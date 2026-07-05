from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PACK_LAYOUT_VERSION = 1

DEFAULT_EXPERT_COMPONENTS: tuple[str, ...] = (
    "gate_proj.weight",
    "gate_proj.scales",
    "gate_proj.biases",
    "up_proj.weight",
    "up_proj.scales",
    "up_proj.biases",
    "down_proj.weight",
    "down_proj.scales",
    "down_proj.biases",
)

MXFP4_EXPERT_COMPONENTS: tuple[str, ...] = (
    "gate_proj.weight",
    "gate_proj.scales",
    "up_proj.weight",
    "up_proj.scales",
    "down_proj.weight",
    "down_proj.scales",
)


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


@dataclass(frozen=True)
class ComponentLayout:
    name: str
    offset: int
    size: int
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class LayerLayout:
    layer: int
    num_experts: int
    expert_slot_bytes: int
    layer_file: str
    components: tuple[ComponentLayout, ...]

    @property
    def layer_file_bytes(self) -> int:
        return self.num_experts * self.expert_slot_bytes


@dataclass(frozen=True)
class PackedExpertsLayout:
    version: int
    model_type: str
    config_sha256: str | None
    quantization: str
    group_size: int | None
    num_layers: int
    num_experts: int
    component_order: tuple[str, ...]
    layers: tuple[LayerLayout, ...]

    @property
    def total_bytes(self) -> int:
        return sum(layer.layer_file_bytes for layer in self.layers)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["component_order"] = list(self.component_order)
        return payload

    def write(self, path: str | Path) -> None:
        p = Path(path)
        _write_json_atomic(p, self.to_json_dict())


def config_sha256(config_path: str | Path) -> str | None:
    p = Path(config_path)
    if p.is_dir():
        p = p / "config.json"
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ResidentTensorLayout:
    name: str
    offset: int
    size: int
    dtype: str
    shape: tuple[int, ...]
    category: str


@dataclass(frozen=True)
class ResidentWeightsLayout:
    version: int
    model_type: str
    config_sha256: str | None
    alignment: int
    weight_file: str
    total_bytes: int
    tensors: tuple[ResidentTensorLayout, ...]
    router: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        p = Path(path)
        _write_json_atomic(p, self.to_json_dict())
