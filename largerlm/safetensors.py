from __future__ import annotations

import json
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SafetensorsError(RuntimeError):
    """Raised when a safetensors index or shard header cannot be read."""


HEADER_MANIFEST_NAME = "largerlm.safetensors.headers.json"
MAX_SAFETENSORS_HEADER_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class TensorMeta:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    data_start: int

    @property
    def nbytes(self) -> int:
        return int(self.data_offsets[1] - self.data_offsets[0])

    @property
    def absolute_start(self) -> int:
        return self.data_start + self.data_offsets[0]


@dataclass(frozen=True)
class CheckpointStats:
    total_bytes: int
    routed_expert_bytes: int
    resident_bytes: int
    unknown_bytes: int
    tensor_count: int
    by_category: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ShardHeaderCheck:
    shard: str
    ok: bool
    data_start: int | None
    file_size: int | None
    tensor_count: int
    error: str | None


ROUTED_EXPERT_PATTERNS = (
    re.compile(r"\.experts\.\d+\."),
    re.compile(
        r"\.experts\."
        r"(gate_proj|up_proj|down_proj|gate_up_proj|gate_up|w1|w2|w3|w13)\."
    ),
    re.compile(
        r"\.switch_mlp\."
        r"(gate_proj|up_proj|down_proj|gate_up_proj|gate_up|w1|w2|w3|w13)\."
    ),
    re.compile(r"\.block_sparse_moe\.experts\."),
)
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.")
_DENSE_MLP_COMPONENT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\..*?\."
    r"(?:switch_mlp\.)?"
    r"(?:gate_proj|up_proj|down_proj|gate_up_proj|gate_up|w1|w2|w3|w13)\."
    r"(?:weight|scales|biases)$"
)
IGNORED_EXTRA_LAYER_CATEGORY = "ignored_extra_layers"
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "UINT8": 1,
    "I8": 1,
    "INT8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "U16": 2,
    "UINT16": 2,
    "I16": 2,
    "INT16": 2,
    "F16": 2,
    "FLOAT16": 2,
    "HALF": 2,
    "BF16": 2,
    "BFLOAT16": 2,
    "U32": 4,
    "UINT32": 4,
    "I32": 4,
    "INT32": 4,
    "F32": 4,
    "FLOAT32": 4,
    "FLOAT": 4,
    "U64": 8,
    "UINT64": 8,
    "I64": 8,
    "INT64": 8,
    "F64": 8,
    "FLOAT64": 8,
    "DOUBLE": 8,
}


def is_routed_expert_tensor(name: str) -> bool:
    if "shared_expert" in name:
        return False
    return any(pattern.search(name) for pattern in ROUTED_EXPERT_PATTERNS)


def categorize_tensor(name: str) -> str:
    if is_routed_expert_tensor(name):
        return "routed_experts"
    if "embed" in name:
        return "embeddings"
    if "lm_head" in name:
        return "lm_head"
    if "shared_expert" in name:
        return "shared_experts"
    if ".gate." in name or name.endswith(".gate.weight"):
        return "routers"
    if "self_attn" in name or "attention" in name or "indexer" in name:
        return "attention"
    if "norm" in name:
        return "norms"
    return "resident_other"


def tensor_layer_id(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group("layer"))


def is_runtime_layer_tensor(name: str, num_hidden_layers: int) -> bool:
    layer = tensor_layer_id(name)
    return layer is None or layer < num_hidden_layers


def is_routed_expert_tensor_for_moe_layers(
    name: str,
    moe_layers: set[int],
    *,
    num_hidden_layers: int | None = None,
) -> bool:
    if num_hidden_layers is not None and not is_runtime_layer_tensor(
        name,
        num_hidden_layers,
    ):
        return False
    layer = tensor_layer_id(name)
    if layer is not None and layer not in moe_layers:
        return False
    return is_routed_expert_tensor(name)


def categorize_tensor_for_moe_layers(
    name: str,
    moe_layers: set[int],
    *,
    num_hidden_layers: int | None = None,
) -> str:
    if num_hidden_layers is not None and not is_runtime_layer_tensor(
        name,
        num_hidden_layers,
    ):
        return IGNORED_EXTRA_LAYER_CATEGORY
    dense_match = _DENSE_MLP_COMPONENT_RE.search(name)
    if dense_match is not None and int(dense_match.group("layer")) not in moe_layers:
        return "dense_mlp"
    if is_routed_expert_tensor_for_moe_layers(
        name,
        moe_layers,
        num_hidden_layers=num_hidden_layers,
    ):
        return "routed_experts"
    if "embed" in name:
        return "embeddings"
    if "lm_head" in name:
        return "lm_head"
    if "shared_expert" in name:
        return "shared_experts"
    if ".gate." in name or name.endswith(".gate.weight"):
        return "routers"
    if "self_attn" in name or "attention" in name or "indexer" in name:
        return "attention"
    if "norm" in name:
        return "norms"
    return "resident_other"


def _parse_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                raise SafetensorsError(f"short safetensors header prefix: {path}")
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len > MAX_SAFETENSORS_HEADER_BYTES:
                raise SafetensorsError(
                    f"safetensors header length {header_len} exceeds "
                    f"{MAX_SAFETENSORS_HEADER_BYTES} byte safety cap: {path}"
                )
            if header_len > max(file_size - 8, 0):
                raise SafetensorsError(
                    f"safetensors header length {header_len} exceeds file "
                    f"payload {max(file_size - 8, 0)} bytes: {path}"
                )
            header_bytes = f.read(header_len)
            if len(header_bytes) != header_len:
                raise SafetensorsError(f"short safetensors header body: {path}")
            header = json.loads(header_bytes)
            if not isinstance(header, dict):
                raise SafetensorsError(f"safetensors header must be an object: {path}")
            return header, 8 + header_len
    except OSError as exc:
        raise SafetensorsError(f"failed to read shard {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SafetensorsError(f"failed to parse safetensors header {path}: {exc}") from exc


def _metadata_int(value: object, *, tensor_name: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SafetensorsError(
            f"invalid safetensors metadata for tensor {tensor_name!r}: "
            f"{field} must be an integer"
        )
    if value < 0:
        raise SafetensorsError(
            f"invalid safetensors metadata for tensor {tensor_name!r}: "
            f"{field} must be non-negative"
        )
    return value


def _expected_tensor_nbytes(
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> int:
    normalized = dtype.upper()
    dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(normalized)
    if dtype_bytes is None:
        raise SafetensorsError(
            f"unsupported safetensors dtype {dtype!r} for tensor {name!r}"
        )
    total = dtype_bytes
    for index, dim in enumerate(shape):
        if dim < 0:
            raise SafetensorsError(
                f"invalid safetensors metadata for tensor {name!r}: "
                f"shape[{index}] must be non-negative"
            )
        total *= dim
    return total


def _validate_shard_name(
    shard: object,
    *,
    tensor_name: str,
) -> str:
    if not isinstance(shard, str) or not shard:
        raise SafetensorsError(
            f"weight_map shard for tensor {tensor_name!r} must be a "
            "non-empty relative path"
        )
    shard_path = Path(shard)
    if shard_path.is_absolute() or any(part == ".." for part in shard_path.parts):
        raise SafetensorsError(
            f"weight_map shard for tensor {tensor_name!r} escapes model directory: "
            f"{shard!r}"
        )
    return shard


def _index_total_size(index_path: Path, metadata: object) -> int | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise SafetensorsError(f"{index_path} metadata must be an object")
    total_size = metadata.get("total_size")
    if total_size is None:
        return None
    if isinstance(total_size, bool) or not isinstance(total_size, int):
        raise SafetensorsError(
            f"{index_path} metadata.total_size must be a non-negative integer"
        )
    if total_size < 0:
        raise SafetensorsError(
            f"{index_path} metadata.total_size must be a non-negative integer"
        )
    return total_size


def _manifest_non_negative_int(
    manifest_path: Path,
    value: object,
    *,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SafetensorsError(f"{manifest_path} {field} must be a non-negative integer")
    return value


def _tensor_meta_from_header(
    *,
    name: str,
    shard: str,
    shard_path: Path,
    header: dict[str, Any],
    data_start: int,
    payload_bytes: int,
) -> TensorMeta:
    meta = header.get(name)
    if not isinstance(meta, dict):
        raise SafetensorsError(
            f"tensor {name!r} is listed in weight_map but missing from {shard}"
        )
    offsets = meta.get("data_offsets")
    shape = meta.get("shape")
    dtype = meta.get("dtype")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not isinstance(shape, list)
        or dtype is None
    ):
        raise SafetensorsError(f"malformed safetensors metadata for tensor {name!r}")
    start = _metadata_int(offsets[0], tensor_name=name, field="data_offsets[0]")
    end = _metadata_int(offsets[1], tensor_name=name, field="data_offsets[1]")
    parsed_shape = tuple(
        _metadata_int(dim, tensor_name=name, field=f"shape[{index}]")
        for index, dim in enumerate(shape)
    )
    if start < 0 or end < start:
        raise SafetensorsError(
            f"invalid data_offsets for tensor {name!r}: [{start}, {end}]"
        )
    if end > payload_bytes:
        raise SafetensorsError(
            f"tensor {name!r} data_offsets end {end} exceeds shard payload "
            f"{payload_bytes} bytes in {shard_path}"
        )
    expected_nbytes = _expected_tensor_nbytes(
        name=name,
        dtype=str(dtype),
        shape=parsed_shape,
    )
    actual_nbytes = end - start
    if actual_nbytes != expected_nbytes:
        raise SafetensorsError(
            f"tensor {name!r} dtype/shape expects {expected_nbytes} bytes "
            f"but data_offsets span {actual_nbytes} bytes"
        )
    return TensorMeta(
        name=name,
        shard=shard,
        dtype=str(dtype),
        shape=parsed_shape,
        data_offsets=(start, end),
        data_start=data_start,
    )


def _validate_shard_tensor_spans(
    shard: str,
    tensors: list[TensorMeta],
    *,
    payload_bytes: int,
) -> None:
    cursor = 0
    for tensor in sorted(
        tensors,
        key=lambda tensor: (
            tensor.data_offsets[0],
            tensor.data_offsets[1],
            tensor.name,
        ),
    ):
        start, end = tensor.data_offsets
        if start < cursor:
            raise SafetensorsError(
                f"shard {shard!r} tensor {tensor.name!r} overlaps previous "
                f"tensor data at offset {start}"
            )
        if start > cursor:
            raise SafetensorsError(
                f"shard {shard!r} has unindexed data gap before tensor "
                f"{tensor.name!r}: expected offset {cursor}, found {start}"
            )
        cursor = end
    if cursor != payload_bytes:
        raise SafetensorsError(
            f"shard {shard!r} has unindexed trailing data: indexed {cursor} "
            f"bytes, payload has {payload_bytes} bytes"
        )


def validate_local_safetensors_header(
    model_dir: str | Path,
    shard: str,
    *,
    index: dict[str, Any] | None = None,
    expected_data_start: int | None = None,
    expected_file_size: int | None = None,
    expected_header: dict[str, Any] | None = None,
) -> ShardHeaderCheck:
    """Validate one local shard using only its safetensors header and stat data."""

    root = Path(model_dir)
    shard_name = _validate_shard_name(shard, tensor_name="<shard>")
    shard_path = root / shard_name
    try:
        header, data_start = _parse_header(shard_path)
        file_size = shard_path.stat().st_size
        if expected_data_start is not None and data_start != expected_data_start:
            raise SafetensorsError(
                f"shard {shard_name!r} data_start {data_start} does not match "
                f"expected {expected_data_start}"
            )
        if expected_file_size is not None and file_size != expected_file_size:
            raise SafetensorsError(
                f"shard {shard_name!r} file size {file_size} does not match "
                f"expected {expected_file_size}"
            )
        if expected_header is not None and header != expected_header:
            raise SafetensorsError(
                f"shard {shard_name!r} local header does not match header manifest"
            )
        payload_bytes = file_size - data_start
        if payload_bytes < 0:
            raise SafetensorsError(
                f"safetensors data start exceeds file size: {shard_path}"
            )
        header_names = {str(name) for name in header if name != "__metadata__"}
        if index is not None:
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                raise SafetensorsError("safetensors index does not contain a weight_map")
            tensor_names = [
                str(name)
                for name, mapped_shard in weight_map.items()
                if _validate_shard_name(mapped_shard, tensor_name=str(name))
                == shard_name
            ]
            indexed_names = set(tensor_names)
            unindexed = tuple(sorted(header_names - indexed_names))
            if unindexed:
                preview = ", ".join(repr(name) for name in unindexed[:4])
                if len(unindexed) > 4:
                    preview += f", +{len(unindexed) - 4} more"
                raise SafetensorsError(
                    f"shard {shard_name!r} contains tensors not listed in "
                    f"weight_map: {preview}"
                )
        else:
            tensor_names = sorted(header_names)
        shard_tensors = [
            _tensor_meta_from_header(
                name=name,
                shard=shard_name,
                shard_path=shard_path,
                header=header,
                data_start=data_start,
                payload_bytes=payload_bytes,
            )
            for name in sorted(tensor_names)
        ]
        _validate_shard_tensor_spans(
            shard_name,
            shard_tensors,
            payload_bytes=payload_bytes,
        )
        return ShardHeaderCheck(
            shard=shard_name,
            ok=True,
            data_start=data_start,
            file_size=file_size,
            tensor_count=len(shard_tensors),
            error=None,
        )
    except (OSError, SafetensorsError) as exc:
        return ShardHeaderCheck(
            shard=shard_name,
            ok=False,
            data_start=None,
            file_size=None,
            tensor_count=0,
            error=str(exc),
        )


def iter_tensor_metadata(
    model_dir: str | Path,
    *,
    prefer_header_manifest: bool = False,
) -> list[TensorMeta]:
    """Read tensor metadata from a Hugging Face safetensors index."""

    root = Path(model_dir)
    header_manifest_path = root / HEADER_MANIFEST_NAME
    local_shards = tuple(sorted(root.glob("*.safetensors")))
    if header_manifest_path.exists() and (prefer_header_manifest or not local_shards):
        return _iter_tensor_metadata_from_header_manifest(root, header_manifest_path)

    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        if not local_shards:
            raise SafetensorsError(f"missing {index_path}")
        if len(local_shards) != 1:
            raise SafetensorsError(
                f"missing {index_path}; sharded safetensors checkpoints must "
                "include an index"
            )
        shard_path = local_shards[0]
        header, data_start = _parse_header(shard_path)
        try:
            payload_bytes = shard_path.stat().st_size - data_start
        except OSError as exc:
            raise SafetensorsError(f"failed to stat shard {shard_path}: {exc}") from exc
        if payload_bytes < 0:
            raise SafetensorsError(
                f"safetensors data start exceeds file size: {shard_path}"
            )
        shard_tensors = [
            _tensor_meta_from_header(
                name=str(name),
                shard=shard_path.name,
                shard_path=shard_path,
                header=header,
                data_start=data_start,
                payload_bytes=payload_bytes,
            )
            for name in sorted(header)
            if str(name) != "__metadata__"
        ]
        _validate_shard_tensor_spans(
            shard_path.name,
            shard_tensors,
            payload_bytes=payload_bytes,
        )
        return shard_tensors

    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        raise SafetensorsError(f"{index_path} must contain a JSON object")
    expected_total_size = _index_total_size(index_path, index.get("metadata"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SafetensorsError(f"{index_path} does not contain a weight_map")

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        tensor_name = str(name)
        shard_name = _validate_shard_name(shard, tensor_name=tensor_name)
        by_shard.setdefault(shard_name, []).append(tensor_name)

    tensors: list[TensorMeta] = []
    for shard, names in sorted(by_shard.items()):
        shard_path = root / shard
        header, data_start = _parse_header(shard_path)
        indexed_names = set(names)
        header_names = {str(name) for name in header if name != "__metadata__"}
        unindexed = tuple(sorted(header_names - indexed_names))
        if unindexed:
            preview = ", ".join(repr(name) for name in unindexed[:4])
            if len(unindexed) > 4:
                preview += f", +{len(unindexed) - 4} more"
            raise SafetensorsError(
                f"shard {shard!r} contains tensors not listed in weight_map: "
                f"{preview}"
            )
        try:
            payload_bytes = shard_path.stat().st_size - data_start
        except OSError as exc:
            raise SafetensorsError(f"failed to stat shard {shard_path}: {exc}") from exc
        if payload_bytes < 0:
            raise SafetensorsError(
                f"safetensors data start exceeds file size: {shard_path}"
            )
        shard_tensors: list[TensorMeta] = []
        for name in sorted(names):
            shard_tensors.append(
                _tensor_meta_from_header(
                    name=name,
                    shard=shard,
                    shard_path=shard_path,
                    header=header,
                    data_start=data_start,
                    payload_bytes=payload_bytes,
                )
            )
        _validate_shard_tensor_spans(
            shard,
            shard_tensors,
            payload_bytes=payload_bytes,
        )
        tensors.extend(shard_tensors)
    actual_total_size = sum(tensor.nbytes for tensor in tensors)
    if (
        expected_total_size is not None
        and actual_total_size != expected_total_size
    ):
        raise SafetensorsError(
            f"{index_path} metadata.total_size {expected_total_size} does not "
            f"match indexed tensor bytes {actual_total_size}"
        )
    return tensors


def _iter_tensor_metadata_from_header_manifest(
    root: Path,
    manifest_path: Path,
) -> list[TensorMeta]:
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except OSError as exc:
        raise SafetensorsError(f"failed to read {manifest_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SafetensorsError(f"failed to parse {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise SafetensorsError(f"{manifest_path} must contain a JSON object")
    version = manifest.get("version")
    if version != 1:
        raise SafetensorsError(f"{manifest_path} version must be 1")
    index = manifest.get("index")
    if not isinstance(index, dict):
        raise SafetensorsError(f"{manifest_path} must contain an index object")
    shards = manifest.get("shards")
    if not isinstance(shards, dict):
        raise SafetensorsError(f"{manifest_path} must contain a shards object")

    expected_total_size = _index_total_size(manifest_path, index.get("metadata"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SafetensorsError(f"{manifest_path} index does not contain a weight_map")

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        tensor_name = str(name)
        shard_name = _validate_shard_name(shard, tensor_name=tensor_name)
        by_shard.setdefault(shard_name, []).append(tensor_name)

    tensors: list[TensorMeta] = []
    for shard, names in sorted(by_shard.items()):
        shard_info = shards.get(shard)
        if not isinstance(shard_info, dict):
            raise SafetensorsError(
                f"{manifest_path} is missing header manifest for shard {shard!r}"
            )
        header = shard_info.get("header")
        if not isinstance(header, dict):
            raise SafetensorsError(
                f"{manifest_path} shard {shard!r} header must be an object"
            )
        data_start = _manifest_non_negative_int(
            manifest_path,
            shard_info.get("data_start"),
            field=f"shards[{shard!r}].data_start",
        )
        file_size = _manifest_non_negative_int(
            manifest_path,
            shard_info.get("file_size"),
            field=f"shards[{shard!r}].file_size",
        )
        if file_size < data_start:
            raise SafetensorsError(
                f"{manifest_path} shard {shard!r} file_size {file_size} is "
                f"smaller than data_start {data_start}"
            )
        payload_bytes = file_size - data_start
        indexed_names = set(names)
        header_names = {str(name) for name in header if name != "__metadata__"}
        unindexed = tuple(sorted(header_names - indexed_names))
        if unindexed:
            preview = ", ".join(repr(name) for name in unindexed[:4])
            if len(unindexed) > 4:
                preview += f", +{len(unindexed) - 4} more"
            raise SafetensorsError(
                f"shard {shard!r} contains tensors not listed in weight_map: "
                f"{preview}"
            )
        shard_tensors: list[TensorMeta] = []
        for name in sorted(names):
            shard_tensors.append(
                _tensor_meta_from_header(
                    name=name,
                    shard=shard,
                    shard_path=root / shard,
                    header=header,
                    data_start=data_start,
                    payload_bytes=payload_bytes,
                )
            )
        _validate_shard_tensor_spans(
            shard,
            shard_tensors,
            payload_bytes=payload_bytes,
        )
        tensors.extend(shard_tensors)

    actual_total_size = sum(tensor.nbytes for tensor in tensors)
    if (
        expected_total_size is not None
        and actual_total_size != expected_total_size
    ):
        raise SafetensorsError(
            f"{manifest_path} metadata.total_size {expected_total_size} does not "
            f"match indexed tensor bytes {actual_total_size}"
        )
    return tensors


def scan_checkpoint(
    model_dir: str | Path,
    *,
    category_fn: Callable[[str], str] | None = None,
    prefer_header_manifest: bool = False,
) -> CheckpointStats:
    tensors = iter_tensor_metadata(
        model_dir,
        prefer_header_manifest=prefer_header_manifest,
    )
    by_category: dict[str, int] = {}
    categorize = category_fn or categorize_tensor
    for tensor in tensors:
        category = categorize(tensor.name)
        by_category[category] = by_category.get(category, 0) + tensor.nbytes

    routed = by_category.get("routed_experts", 0)
    ignored = sum(
        size
        for category, size in by_category.items()
        if category.startswith("ignored_")
    )
    total = sum(t.nbytes for t in tensors)
    resident = total - routed - ignored
    return CheckpointStats(
        total_bytes=total,
        routed_expert_bytes=routed,
        resident_bytes=resident,
        unknown_bytes=by_category.get("resident_other", 0),
        tensor_count=len(tensors),
        by_category=dict(sorted(by_category.items())),
    )
