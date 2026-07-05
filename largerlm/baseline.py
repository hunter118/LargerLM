from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BASELINE_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class BaselineError(RuntimeError):
    """Raised when a baseline manifest or tensor payload is invalid."""


@dataclass(frozen=True)
class BaselineTensor:
    name: str
    path: str
    dtype: str
    shape: tuple[int, ...]
    bytes: int
    sha256: str


@dataclass(frozen=True)
class BaselineRecord:
    kind: str
    token_index: int
    layer: int | None
    tensors: tuple[BaselineTensor, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineManifest:
    version: int
    model_type: str
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    records: tuple[BaselineRecord, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineValidation:
    root: Path
    tensor_count: int
    total_bytes: int
    ok: bool


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or "tensor"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    except OSError:
        _remove_partial_file(tmp_path)
        raise


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        _remove_partial_file(tmp_path)
        raise


def load_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise BaselineError(f"missing baseline manifest: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as exc:
        raise BaselineError(f"failed to parse baseline manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BaselineError("baseline manifest must be a JSON object")
    version = manifest.get("version")
    if version != BASELINE_VERSION:
        raise BaselineError(f"unsupported baseline version {version}")
    _validate_manifest_shape(manifest)
    return manifest


def iter_manifest_tensors(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tensors: list[dict[str, Any]] = []
    records = manifest.get("records")
    if not isinstance(records, list):
        raise BaselineError("baseline manifest records must be an array")
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise BaselineError(f"baseline record {record_index} must be an object")
        raw_tensors = record.get("tensors")
        if not isinstance(raw_tensors, list):
            raise BaselineError(
                f"baseline record {record_index} tensors must be an array"
            )
        for tensor_index, tensor in enumerate(raw_tensors):
            if not isinstance(tensor, dict):
                raise BaselineError(
                    f"baseline record {record_index} tensor {tensor_index} "
                    "must be an object"
                )
            tensors.append(tensor)
    return tensors


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BaselineError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise BaselineError(f"{label} must be a non-negative integer")
    return int(value)


def _require_token_sequence(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise BaselineError(f"{label} must be an array")
    for index, token in enumerate(value):
        _require_nonnegative_int(token, f"{label}[{index}]")


def _require_shape(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise BaselineError(f"{label} shape must be an array")
    for index, dim in enumerate(value):
        if type(dim) is not int or dim <= 0:
            raise BaselineError(f"{label} shape[{index}] must be a positive integer")


def _validate_tensor_metadata(tensor: dict[str, Any], label: str) -> None:
    _require_nonempty_string(tensor.get("name"), f"{label} name")
    _require_nonempty_string(tensor.get("dtype"), f"{label} dtype")
    _require_shape(tensor.get("shape"), label)
    _require_nonnegative_int(tensor.get("bytes"), f"{label} bytes")
    digest = _require_nonempty_string(tensor.get("sha256"), f"{label} sha256")
    if _SHA256_RE.match(digest) is None:
        raise BaselineError(f"{label} sha256 must be a 64-character hex digest")
    rel = _require_nonempty_string(tensor.get("path"), f"{label} path")
    path = Path(rel)
    if path.is_absolute() or rel in {".", ""} or ".." in path.parts:
        raise BaselineError(f"invalid tensor path in manifest: {rel!r}")


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    _require_nonempty_string(manifest.get("model_type"), "model_type")
    _require_token_sequence(manifest.get("prompt_tokens"), "prompt_tokens")
    _require_token_sequence(manifest.get("generated_tokens"), "generated_tokens")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise BaselineError("baseline manifest metadata must be an object")
    records = manifest.get("records")
    if not isinstance(records, list):
        raise BaselineError("baseline manifest records must be an array")
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise BaselineError(f"baseline record {record_index} must be an object")
        prefix = f"baseline record {record_index}"
        _require_nonempty_string(record.get("kind"), f"{prefix} kind")
        _require_nonnegative_int(record.get("token_index"), f"{prefix} token_index")
        layer = record.get("layer")
        if layer is not None:
            _require_nonnegative_int(layer, f"{prefix} layer")
        record_metadata = record.get("metadata", {})
        if not isinstance(record_metadata, dict):
            raise BaselineError(f"{prefix} metadata must be an object")
        raw_tensors = record.get("tensors")
        if not isinstance(raw_tensors, list):
            raise BaselineError(f"{prefix} tensors must be an array")
        for tensor_index, tensor in enumerate(raw_tensors):
            if not isinstance(tensor, dict):
                raise BaselineError(
                    f"{prefix} tensor {tensor_index} must be an object"
                )
            _validate_tensor_metadata(tensor, f"{prefix} tensor {tensor_index}")


def validate_baseline(root: str | Path) -> BaselineValidation:
    baseline_root = Path(root)
    manifest = load_manifest(baseline_root)
    total = 0
    count = 0
    for tensor in iter_manifest_tensors(manifest):
        rel = str(tensor["path"])
        path = baseline_root / rel
        if not path.exists():
            raise BaselineError(f"missing baseline tensor: {path}")
        data = path.read_bytes()
        expected_bytes = int(tensor["bytes"])
        if len(data) != expected_bytes:
            raise BaselineError(
                f"tensor {rel} has {len(data)} bytes, expected {expected_bytes}"
            )
        expected_hash = str(tensor["sha256"])
        actual_hash = _sha256(data)
        if actual_hash != expected_hash:
            raise BaselineError(f"tensor {rel} sha256 mismatch")
        total += len(data)
        count += 1
    return BaselineValidation(
        root=baseline_root,
        tensor_count=count,
        total_bytes=total,
        ok=True,
    )


class BaselineWriter:
    """Write a lightweight raw-tensor baseline for MLX-vs-Metal checks."""

    def __init__(
        self,
        root: str | Path,
        *,
        model_type: str,
        prompt_tokens: list[int] | tuple[int, ...],
        metadata: dict[str, Any] | None = None,
    ):
        self.root = Path(root)
        self.tensor_dir = self.root / "tensors"
        self.model_type = model_type
        self.prompt_tokens = tuple(int(t) for t in prompt_tokens)
        self.generated_tokens: list[int] = []
        self.metadata = dict(metadata or {})
        self.records: list[BaselineRecord] = []
        self._tensor_counter = 0

    def __enter__(self) -> "BaselineWriter":
        self.root.mkdir(parents=True, exist_ok=True)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.write_manifest()

    def add_generated_token(self, token: int) -> None:
        self.generated_tokens.append(int(token))

    def write_tensor(
        self,
        *,
        name: str,
        dtype: str,
        shape: list[int] | tuple[int, ...],
        data: bytes,
        max_bytes: int | None = None,
    ) -> BaselineTensor:
        if max_bytes is not None and len(data) > max_bytes:
            raise BaselineError(
                f"tensor {name} has {len(data)} bytes, exceeding limit {max_bytes}"
            )
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        self._tensor_counter += 1
        filename = f"{self._tensor_counter:06d}_{_safe_name(name)}.bin"
        path = self.tensor_dir / filename
        _write_bytes_atomic(path, data)
        return BaselineTensor(
            name=name,
            path=str(path.relative_to(self.root)),
            dtype=dtype,
            shape=tuple(int(x) for x in shape),
            bytes=len(data),
            sha256=_sha256(data),
        )

    def add_record(
        self,
        *,
        kind: str,
        token_index: int,
        layer: int | None,
        tensors: list[BaselineTensor] | tuple[BaselineTensor, ...],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.records.append(
            BaselineRecord(
                kind=kind,
                token_index=int(token_index),
                layer=None if layer is None else int(layer),
                tensors=tuple(tensors),
                metadata=dict(metadata or {}),
            )
        )

    def manifest(self) -> BaselineManifest:
        return BaselineManifest(
            version=BASELINE_VERSION,
            model_type=self.model_type,
            prompt_tokens=self.prompt_tokens,
            generated_tokens=tuple(self.generated_tokens),
            records=tuple(self.records),
            metadata=self.metadata,
        )

    def write_manifest(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "manifest.json"
        _write_json_atomic(path, self.manifest().to_json_dict())
        return path
