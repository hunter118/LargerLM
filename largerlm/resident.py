from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from .config import ModelConfig, load_config
from .layout import ResidentTensorLayout, ResidentWeightsLayout, config_sha256
from .safety import (
    DiskBudget,
    check_chunk_budget,
    check_disk_budget,
    check_memory_budget,
    disk_budget,
    estimate_pack_peak_heap_bytes,
)
from .safetensors import (
    IGNORED_EXTRA_LAYER_CATEGORY,
    TensorMeta,
    categorize_tensor_for_moe_layers,
    iter_tensor_metadata,
    is_runtime_layer_tensor,
)


class ResidentPackerError(RuntimeError):
    """Raised when resident weights cannot be packed safely."""


FUSED_GATE_UP_COMPONENTS = ("gate_up_proj", "gate_up", "w13")
RESIDENT_MLP_COMPONENT_ALIASES = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}
AFFINE_INT4_WEIGHT_DTYPES = {"U32", "UINT32"}
AFFINE_INT4_META_DTYPES = {"BF16", "BFLOAT16", "F16", "FLOAT16"}
MXFP4_SCALE_DTYPES = {"U8", "UINT8"}


@dataclass(frozen=True)
class ResidentAliasExpansionStats:
    component_alias_source_tensor_count: int = 0
    component_alias_renamed_tensor_count: int = 0
    component_alias_bytes: int = 0
    fused_gate_up_source_tensor_count: int = 0
    fused_gate_up_expanded_tensor_count: int = 0
    fused_gate_up_expanded_bytes: int = 0


@dataclass(frozen=True)
class ResidentPackReport:
    layout: ResidentWeightsLayout
    output_dir: Path
    dry_run: bool
    chunk_size: int
    estimated_peak_heap_bytes: int
    max_heap_bytes: int
    disk_budget: DiskBudget
    disk_checked: bool
    component_alias_source_tensor_count: int = 0
    component_alias_renamed_tensor_count: int = 0
    component_alias_bytes: int = 0
    fused_gate_up_source_tensor_count: int = 0
    fused_gate_up_expanded_tensor_count: int = 0
    fused_gate_up_expanded_bytes: int = 0


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _router_metadata(cfg: ModelConfig) -> dict[str, object] | None:
    payload: dict[str, object] = {}
    if cfg.scoring_func is not None:
        payload["scoring_func"] = cfg.scoring_func
    if cfg.norm_topk_prob is not None:
        payload["norm_topk_prob"] = cfg.norm_topk_prob
    if cfg.routed_scaling_factor is not None:
        payload["routed_scaling_factor"] = cfg.routed_scaling_factor
    if cfg.n_group is not None:
        payload["n_group"] = cfg.n_group
    if cfg.topk_group is not None:
        payload["topk_group"] = cfg.topk_group
    if cfg.topk_method is not None:
        payload["topk_method"] = cfg.topk_method
    if cfg.num_experts_per_tok is not None:
        payload["num_experts_per_tok"] = int(cfg.num_experts_per_tok)
    return payload or None


def _replace_fused_gate_up_component(name: str, replacement: str) -> str | None:
    for component in FUSED_GATE_UP_COMPONENTS:
        suffix = f".{component}.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".{replacement}.weight"
    return None


def _replace_resident_mlp_component_alias(name: str) -> str | None:
    for alias, replacement in RESIDENT_MLP_COMPONENT_ALIASES.items():
        suffix = f".{alias}.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".{replacement}.weight"
    return None


def _split_fused_gate_up_resident_tensor(
    tensor: TensorMeta,
    *,
    split_rows: int,
    hidden_size: int,
) -> tuple[TensorMeta, TensorMeta]:
    gate_name = _replace_fused_gate_up_component(tensor.name, "gate_proj")
    up_name = _replace_fused_gate_up_component(tensor.name, "up_proj")
    if gate_name is None or up_name is None:
        raise ResidentPackerError(
            f"resident tensor {tensor.name} is not a fused gate/up tensor"
        )
    expected_shape = (2 * split_rows, hidden_size)
    if tensor.shape != expected_shape:
        raise ResidentPackerError(
            f"resident fused gate/up tensor {tensor.name} shape "
            f"{tensor.shape} does not match expected {expected_shape}"
        )
    if tensor.nbytes % 2 != 0:
        raise ResidentPackerError(
            f"resident fused gate/up tensor {tensor.name} byte size "
            f"{tensor.nbytes} is not divisible into gate/up halves"
        )
    half_bytes = tensor.nbytes // 2
    start, end = tensor.data_offsets
    middle = start + half_bytes
    if middle > end:
        raise ResidentPackerError(
            f"resident fused gate/up tensor {tensor.name} split exceeds source span"
        )
    split_shape = (split_rows, hidden_size)
    return (
        TensorMeta(
            name=gate_name,
            shard=tensor.shard,
            dtype=tensor.dtype,
            shape=split_shape,
            data_offsets=(start, middle),
            data_start=tensor.data_start,
        ),
        TensorMeta(
            name=up_name,
            shard=tensor.shard,
            dtype=tensor.dtype,
            shape=split_shape,
            data_offsets=(middle, end),
            data_start=tensor.data_start,
        ),
    )


def _resident_fused_gate_up_split_shape(
    cfg: ModelConfig,
    *,
    category: str,
) -> tuple[int, int] | None:
    hidden = int(cfg.hidden_size)
    if category == "dense_mlp":
        if cfg.intermediate_size is None:
            return None
        return int(cfg.intermediate_size), hidden
    if category == "shared_experts":
        shared_count = int(cfg.n_shared_experts or 0)
        if shared_count <= 0:
            return None
        return shared_count * int(cfg.moe_hidden_size), hidden
    return None


def _normalize_resident_tensor_aliases_with_stats(
    tensors: list[TensorMeta],
    *,
    config: ModelConfig,
) -> tuple[list[TensorMeta], ResidentAliasExpansionStats]:
    moe_layers = set(config.moe_layers)
    normalized: list[TensorMeta] = []
    component_alias_source_count = 0
    component_alias_renamed_count = 0
    component_alias_bytes = 0
    fused_gate_up_source_count = 0
    fused_gate_up_expanded_count = 0
    fused_gate_up_expanded_bytes = 0
    for tensor in tensors:
        category = categorize_tensor_for_moe_layers(
            tensor.name,
            moe_layers,
            num_hidden_layers=config.num_hidden_layers,
        )
        if category in {"dense_mlp", "shared_experts"}:
            alias_name = _replace_resident_mlp_component_alias(tensor.name)
            if alias_name is not None:
                tensor = replace(tensor, name=alias_name)
                component_alias_source_count += 1
                component_alias_renamed_count += 1
                component_alias_bytes += tensor.nbytes
        split_shape = _resident_fused_gate_up_split_shape(
            config,
            category=category,
        )
        if (
            split_shape is not None
            and _replace_fused_gate_up_component(tensor.name, "gate_proj") is not None
        ):
            split_rows, hidden_size = split_shape
            split = _split_fused_gate_up_resident_tensor(
                tensor,
                split_rows=split_rows,
                hidden_size=hidden_size,
            )
            normalized.extend(split)
            fused_gate_up_source_count += 1
            fused_gate_up_expanded_count += len(split)
            fused_gate_up_expanded_bytes += sum(item.nbytes for item in split)
        else:
            normalized.append(tensor)

    seen: set[str] = set()
    for tensor in normalized:
        if tensor.name in seen:
            raise ResidentPackerError(
                f"duplicate resident tensor {tensor.name} after alias expansion"
            )
        seen.add(tensor.name)
    return (
        normalized,
        ResidentAliasExpansionStats(
            component_alias_source_tensor_count=component_alias_source_count,
            component_alias_renamed_tensor_count=component_alias_renamed_count,
            component_alias_bytes=component_alias_bytes,
            fused_gate_up_source_tensor_count=fused_gate_up_source_count,
            fused_gate_up_expanded_tensor_count=fused_gate_up_expanded_count,
            fused_gate_up_expanded_bytes=fused_gate_up_expanded_bytes,
        ),
    )


def normalize_resident_tensor_aliases(
    tensors: list[TensorMeta],
    *,
    config: ModelConfig,
) -> list[TensorMeta]:
    normalized, _stats = _normalize_resident_tensor_aliases_with_stats(
        tensors,
        config=config,
    )
    return normalized


def _is_affine_int4_weight_dtype(dtype: str) -> bool:
    return dtype.upper() in AFFINE_INT4_WEIGHT_DTYPES


def _is_affine_int4_meta_dtype(dtype: str) -> bool:
    return dtype.upper() in AFFINE_INT4_META_DTYPES


def _is_mxfp4_scale_dtype(dtype: str) -> bool:
    return dtype.upper() in MXFP4_SCALE_DTYPES


def resident_mxfp4_logical_shape(
    tensor: TensorMeta,
    tensors_by_name: dict[str, TensorMeta],
) -> tuple[int, ...] | None:
    if not _is_affine_int4_weight_dtype(tensor.dtype):
        return None
    if not tensor.name.endswith(".weight"):
        return None
    base = tensor.name[: -len(".weight")]
    scales = tensors_by_name.get(f"{base}.scales")
    biases = tensors_by_name.get(f"{base}.biases")
    if scales is None or biases is not None:
        return None
    if not _is_mxfp4_scale_dtype(scales.dtype):
        return None
    if len(tensor.shape) < 2:
        raise ResidentPackerError(
            f"resident MXFP4 weight {tensor.name} must be at least 2-D"
        )
    if len(scales.shape) != len(tensor.shape):
        raise ResidentPackerError(
            f"resident MXFP4 scales for {tensor.name} must match weight rank"
        )
    if any(dim <= 0 for dim in tensor.shape):
        raise ResidentPackerError(
            f"resident MXFP4 weight {tensor.name} dimensions must be positive"
        )
    if any(dim <= 0 for dim in scales.shape):
        raise ResidentPackerError(
            f"resident MXFP4 scales for {tensor.name} dimensions must be positive"
        )
    if scales.shape[:-1] != tensor.shape[:-1]:
        raise ResidentPackerError(
            f"resident MXFP4 scales for {tensor.name} must match weight prefix shape"
        )
    packed_cols = tensor.shape[-1]
    groups = scales.shape[-1]
    in_dim = packed_cols * 8
    if groups <= 0 or in_dim % groups != 0:
        raise ResidentPackerError(
            f"resident MXFP4 groups for {tensor.name} do not divide "
            f"logical input dim {in_dim}"
        )
    group_size = in_dim // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise ResidentPackerError(
            f"resident MXFP4 group size for {tensor.name} must be a "
            f"positive multiple of 8, got {group_size}"
        )
    expected_weight_bytes = 4
    for dim in tensor.shape:
        expected_weight_bytes *= int(dim)
    expected_scale_bytes = 1
    for dim in scales.shape:
        expected_scale_bytes *= int(dim)
    if tensor.nbytes != expected_weight_bytes:
        raise ResidentPackerError(
            f"resident MXFP4 weight {tensor.name} size does not match packed shape"
        )
    if scales.nbytes != expected_scale_bytes:
        raise ResidentPackerError(
            f"resident MXFP4 scales for {tensor.name} size does not match metadata shape"
        )
    return tuple(int(dim) for dim in tensor.shape[:-1]) + (int(in_dim),)


def resident_affine_int4_logical_shape(
    tensor: TensorMeta,
    tensors_by_name: dict[str, TensorMeta],
) -> tuple[int, ...] | None:
    if not _is_affine_int4_weight_dtype(tensor.dtype):
        return None
    if not tensor.name.endswith(".weight"):
        return None
    mxfp4_shape = resident_mxfp4_logical_shape(tensor, tensors_by_name)
    if mxfp4_shape is not None:
        return mxfp4_shape
    if len(tensor.shape) != 2:
        raise ResidentPackerError(
            f"resident affine-int4 weight {tensor.name} must be 2-D"
        )
    base = tensor.name[: -len(".weight")]
    scales = tensors_by_name.get(f"{base}.scales")
    biases = tensors_by_name.get(f"{base}.biases")
    if scales is None or biases is None:
        raise ResidentPackerError(
            f"resident affine-int4 weight {tensor.name} is missing scales/biases "
            "companions"
        )
    if not _is_affine_int4_meta_dtype(scales.dtype) or not _is_affine_int4_meta_dtype(
        biases.dtype
    ):
        raise ResidentPackerError(
            f"resident affine-int4 metadata for {tensor.name} must be BF16/F16"
        )
    if len(scales.shape) != 2 or len(biases.shape) != 2:
        raise ResidentPackerError(
            f"resident affine-int4 metadata for {tensor.name} must be 2-D"
        )
    out_dim, packed_cols = tensor.shape
    if out_dim <= 0 or packed_cols <= 0:
        raise ResidentPackerError(
            f"resident affine-int4 weight {tensor.name} dimensions must be positive"
        )
    if scales.shape != biases.shape or scales.shape[0] != out_dim:
        raise ResidentPackerError(
            f"resident affine-int4 metadata for {tensor.name} must have shape "
            "[out_dim, groups]"
        )
    groups = scales.shape[1]
    in_dim = packed_cols * 8
    if groups <= 0 or in_dim % groups != 0:
        raise ResidentPackerError(
            f"resident affine-int4 groups for {tensor.name} do not divide "
            f"logical input dim {in_dim}"
        )
    group_size = in_dim // groups
    if group_size <= 0 or group_size % 8 != 0:
        raise ResidentPackerError(
            f"resident affine-int4 group size for {tensor.name} must be a "
            f"positive multiple of 8, got {group_size}"
        )
    return int(out_dim), int(in_dim)


def validate_resident_affine_int4_tensors(
    tensors: list[TensorMeta],
    *,
    config: ModelConfig,
) -> None:
    tensors_by_name = {tensor.name: tensor for tensor in tensors}
    moe_layers = set(config.moe_layers)
    for tensor in tensors:
        name = tensor.name
        if not _is_affine_int4_weight_dtype(tensor.dtype):
            continue
        if not name.endswith(".weight"):
            continue
        category = categorize_tensor_for_moe_layers(
            name,
            moe_layers,
            num_hidden_layers=config.num_hidden_layers,
        )
        if category in {"routed_experts", IGNORED_EXTRA_LAYER_CATEGORY}:
            continue
        if not is_runtime_layer_tensor(name, config.num_hidden_layers):
            continue
        resident_affine_int4_logical_shape(tensor, tensors_by_name)


def _build_resident_layout_with_stats(
    model_dir: str | Path,
    *,
    alignment: int = 64,
    prefer_header_manifest: bool = False,
) -> tuple[ResidentWeightsLayout, list[TensorMeta], ResidentAliasExpansionStats]:
    cfg = load_config(model_dir)
    moe_layers = set(cfg.moe_layers)
    normalized_tensors, alias_stats = _normalize_resident_tensor_aliases_with_stats(
        iter_tensor_metadata(
            model_dir,
            prefer_header_manifest=prefer_header_manifest,
        ),
        config=cfg,
    )
    validate_resident_affine_int4_tensors(
        normalized_tensors,
        config=cfg,
    )
    categorized = [
        (
            tensor,
            categorize_tensor_for_moe_layers(
                tensor.name,
                moe_layers,
                num_hidden_layers=cfg.num_hidden_layers,
            ),
        )
        for tensor in normalized_tensors
    ]
    tensors = [
        tensor
        for tensor, category in categorized
        if category not in {"routed_experts", IGNORED_EXTRA_LAYER_CATEGORY}
    ]
    tensors.sort(key=lambda tensor: tensor.name)
    if not tensors:
        raise ResidentPackerError("no resident tensors found")

    offset = 0
    layouts: list[ResidentTensorLayout] = []
    for tensor in tensors:
        offset = _align(offset, alignment)
        layouts.append(
            ResidentTensorLayout(
                name=tensor.name,
                offset=offset,
                size=tensor.nbytes,
                dtype=tensor.dtype,
                shape=tensor.shape,
                category=categorize_tensor_for_moe_layers(
                    tensor.name,
                    moe_layers,
                    num_hidden_layers=cfg.num_hidden_layers,
                ),
            )
        )
        offset += tensor.nbytes

    total = _align(offset, alignment)
    return (
        ResidentWeightsLayout(
            version=1,
            model_type=cfg.model_type,
            config_sha256=config_sha256(model_dir),
            alignment=alignment,
            weight_file="resident.bin",
            total_bytes=total,
            tensors=tuple(layouts),
            router=_router_metadata(cfg),
        ),
        tensors,
        alias_stats,
    )


def build_resident_layout(
    model_dir: str | Path,
    *,
    alignment: int = 64,
    prefer_header_manifest: bool = False,
) -> tuple[ResidentWeightsLayout, list[TensorMeta]]:
    layout, tensors, _alias_stats = _build_resident_layout_with_stats(
        model_dir,
        alignment=alignment,
        prefer_header_manifest=prefer_header_manifest,
    )
    return layout, tensors


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
            raise ResidentPackerError(
                f"short read at offset {src_offset + copied}: "
                f"expected {to_read}, got {len(data)}"
            )
        written = os.pwrite(dst_fd, data, dst_offset + copied)
        if written != len(data):
            raise ResidentPackerError(
                f"short write at offset {dst_offset + copied}: "
                f"expected {len(data)}, got {written}"
            )
        copied += to_read


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def pack_resident_weights(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = True,
    force: bool = False,
    chunk_size: int = 8 * 1024**2,
    max_chunk_size: int = 64 * 1024**2,
    max_heap_bytes: int = 512 * 1024**2,
    disk_safety_margin_bytes: int = 16 * 1024**3,
    alignment: int = 64,
    prefer_header_manifest: bool = False,
) -> ResidentPackReport:
    root = Path(model_dir)
    out = Path(output_dir)
    if prefer_header_manifest and not dry_run:
        raise ResidentPackerError(
            "prefer_header_manifest is only supported for dry-run packing"
        )
    check_chunk_budget(chunk_size, max_chunk_size)
    estimated_peak_heap = estimate_pack_peak_heap_bytes(chunk_size=chunk_size)
    check_memory_budget(estimated_peak_heap, max_heap_bytes)
    layout, tensors, alias_stats = _build_resident_layout_with_stats(
        root,
        alignment=alignment,
        prefer_header_manifest=prefer_header_manifest,
    )
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
        weight_path = out / layout.weight_file
        if layout_path.exists() and not force:
            raise ResidentPackerError(f"{layout_path} already exists; use --force")
        if weight_path.exists() and not force:
            raise ResidentPackerError(f"{weight_path} already exists; use --force")

        fd_out = os.open(weight_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        open_fds: dict[str, int] = {}
        partial_paths = [weight_path]
        try:
            os.ftruncate(fd_out, layout.total_bytes)
            tensor_by_name = {tensor.name: tensor for tensor in tensors}
            for tensor_layout in layout.tensors:
                tensor = tensor_by_name[tensor_layout.name]
                shard_path = str(root / tensor.shard)
                if shard_path not in open_fds:
                    open_fds[shard_path] = os.open(shard_path, os.O_RDONLY)
                _copy_slice(
                    src_fd=open_fds[shard_path],
                    dst_fd=fd_out,
                    src_offset=tensor.absolute_start,
                    dst_offset=tensor_layout.offset,
                    size=tensor_layout.size,
                    chunk_size=chunk_size,
                )
            layout.write(layout_path)
        except OSError as exc:
            for path in reversed(partial_paths):
                _remove_partial_file(path)
            raise ResidentPackerError(f"failed to pack resident weights: {exc}") from exc
        except ResidentPackerError:
            for path in reversed(partial_paths):
                _remove_partial_file(path)
            raise
        finally:
            os.close(fd_out)
            for fd in open_fds.values():
                os.close(fd)

    return ResidentPackReport(
        layout=layout,
        output_dir=out,
        dry_run=dry_run,
        chunk_size=chunk_size,
        estimated_peak_heap_bytes=estimated_peak_heap,
        max_heap_bytes=max_heap_bytes,
        disk_budget=budget,
        disk_checked=disk_checked,
        component_alias_source_tensor_count=(
            alias_stats.component_alias_source_tensor_count
        ),
        component_alias_renamed_tensor_count=(
            alias_stats.component_alias_renamed_tensor_count
        ),
        component_alias_bytes=alias_stats.component_alias_bytes,
        fused_gate_up_source_tensor_count=(
            alias_stats.fused_gate_up_source_tensor_count
        ),
        fused_gate_up_expanded_tensor_count=(
            alias_stats.fused_gate_up_expanded_tensor_count
        ),
        fused_gate_up_expanded_bytes=alias_stats.fused_gate_up_expanded_bytes,
    )
