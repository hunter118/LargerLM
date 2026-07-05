#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from largerlm.config import ConfigError, load_config


GIB = 1024**3

FLASH_MOE_QWEN_REFERENCE = {
    "name": "flash-moe-qwen3.5-397b-a17b-q4",
    "source": "danveloper/flash-moe README and optimization notes",
    "num_layers": 60,
    "moe_layer_count": 60,
    "top_k": 4,
    "expert_slot_bytes": 7_077_888,
    "reported_tokens_per_second": 4.36,
}


class EfficiencyEnvelopeError(RuntimeError):
    """Raised when the static efficiency envelope cannot be computed."""


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EfficiencyEnvelopeError(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EfficiencyEnvelopeError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EfficiencyEnvelopeError(f"{path} must contain a JSON object")
    return payload


def _resolve_manifest_path(prepared_dir: Path) -> Path:
    manifest_path = prepared_dir / "manifest.json"
    if not manifest_path.exists():
        raise EfficiencyEnvelopeError(f"prepared manifest not found: {manifest_path}")
    return manifest_path


def _resolve_relative_path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EfficiencyEnvelopeError(f"manifest is missing {label}")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EfficiencyEnvelopeError(f"{label} must be numeric") from exc
    if parsed <= 0.0:
        raise EfficiencyEnvelopeError(f"{label} must be positive")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EfficiencyEnvelopeError(f"{label} must be an integer") from exc
    if parsed <= 0:
        raise EfficiencyEnvelopeError(f"{label} must be positive")
    return parsed


def _expert_layout_summary(layout_path: Path) -> dict[str, Any]:
    layout = _load_json_object(layout_path)
    raw_layers = layout.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise EfficiencyEnvelopeError(f"{layout_path} has no expert layers")

    slot_bytes_by_layer: list[int] = []
    layer_ids: list[int] = []
    num_experts_by_layer: list[int] = []
    for index, item in enumerate(raw_layers):
        if not isinstance(item, dict):
            raise EfficiencyEnvelopeError(
                f"{layout_path} layer entry {index} must be an object"
            )
        layer_id = _positive_int(item.get("layer"), f"layer entry {index}.layer")
        slot_bytes = _positive_int(
            item.get("expert_slot_bytes"),
            f"layer {layer_id}.expert_slot_bytes",
        )
        num_experts = _positive_int(
            item.get("num_experts"),
            f"layer {layer_id}.num_experts",
        )
        layer_ids.append(layer_id)
        slot_bytes_by_layer.append(slot_bytes)
        num_experts_by_layer.append(num_experts)

    return {
        "layout_path": str(layout_path),
        "moe_layer_count": len(slot_bytes_by_layer),
        "layer_ids": layer_ids,
        "min_expert_slot_bytes": min(slot_bytes_by_layer),
        "max_expert_slot_bytes": max(slot_bytes_by_layer),
        "total_one_expert_per_layer_bytes": sum(slot_bytes_by_layer),
        "min_num_experts": min(num_experts_by_layer),
        "max_num_experts": max(num_experts_by_layer),
        "expert_quantization": layout.get("expert_quantization")
        or layout.get("quantization"),
        "group_size": layout.get("group_size"),
    }


def _io_floor(bytes_per_token: int, io_gib_per_second: float) -> dict[str, Any]:
    gib_per_token = bytes_per_token / GIB
    seconds = gib_per_token / io_gib_per_second
    return {
        "bytes_per_token": bytes_per_token,
        "gib_per_token": gib_per_token,
        "io_gib_per_second": io_gib_per_second,
        "io_floor_seconds_per_token": seconds,
        "io_floor_tokens_per_second": (1.0 / seconds) if seconds > 0.0 else None,
    }


def _reference_envelope(io_gib_per_second: float) -> dict[str, Any]:
    bytes_per_token = (
        int(FLASH_MOE_QWEN_REFERENCE["moe_layer_count"])
        * int(FLASH_MOE_QWEN_REFERENCE["top_k"])
        * int(FLASH_MOE_QWEN_REFERENCE["expert_slot_bytes"])
    )
    floor = _io_floor(bytes_per_token, io_gib_per_second)
    reported_tps = float(FLASH_MOE_QWEN_REFERENCE["reported_tokens_per_second"])
    return {
        **FLASH_MOE_QWEN_REFERENCE,
        **floor,
        "reported_seconds_per_token": 1.0 / reported_tps,
    }


def build_efficiency_envelope(
    prepared_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    expert_layout_path: str | Path | None = None,
    top_k: int | None = None,
    io_gib_per_second: float | None = None,
) -> dict[str, Any]:
    prepared = Path(prepared_dir)
    manifest_path = _resolve_manifest_path(prepared)
    manifest = _load_json_object(manifest_path)

    resolved_layout = (
        Path(expert_layout_path)
        if expert_layout_path is not None
        else _resolve_relative_path(prepared, manifest.get("experts_layout"), "experts_layout")
    )
    layout_summary = _expert_layout_summary(resolved_layout)

    resolved_config = Path(config_path) if config_path is not None else None
    if resolved_config is None:
        model_dir = _resolve_relative_path(prepared, manifest.get("model_dir"), "model_dir")
        resolved_config = model_dir / "config.json"
    try:
        config = load_config(resolved_config)
    except ConfigError as exc:
        raise EfficiencyEnvelopeError(f"failed to load model config: {exc}") from exc

    effective_top_k = top_k if top_k is not None else config.experts_per_token
    effective_top_k = _positive_int(effective_top_k, "top_k")
    if effective_top_k > int(layout_summary["min_num_experts"]):
        raise EfficiencyEnvelopeError(
            "top_k exceeds the minimum expert count in the expert layout"
        )

    effective_io_gib = (
        _positive_float(io_gib_per_second, "io_gib_per_second")
        if io_gib_per_second is not None
        else _positive_float(
            manifest.get("prepare_cold_read_gib_per_second"),
            "manifest.prepare_cold_read_gib_per_second",
        )
    )
    io_source = (
        "argument"
        if io_gib_per_second is not None
        else "manifest.prepare_cold_read_gib_per_second"
    )

    one_route_bytes = int(layout_summary["total_one_expert_per_layer_bytes"])
    bytes_per_token = one_route_bytes * effective_top_k
    glm_floor = _io_floor(bytes_per_token, effective_io_gib)
    reference = _reference_envelope(effective_io_gib)

    ratio = bytes_per_token / int(reference["bytes_per_token"])
    reference_reported_tps = float(reference["reported_tokens_per_second"])
    if glm_floor["io_floor_seconds_per_token"]:
        same_efficiency_tps = reference_reported_tps / ratio
    else:
        same_efficiency_tps = None

    return {
        "schema": "largerlm.flash_moe_efficiency_envelope.v1",
        "prepared_dir": str(prepared),
        "manifest_path": str(manifest_path),
        "model": {
            "config_path": str(resolved_config),
            "model_type": config.model_type,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "moe_layer_count_from_config": config.num_moe_layers,
            "top_k": effective_top_k,
            "expert_quantization": manifest.get("expert_quantization"),
            "expert_group_size": manifest.get("expert_group_size"),
        },
        "expert_layout": layout_summary,
        "routed_expert_io": {
            **glm_floor,
            "io_gib_per_second_source": io_source,
            "total_one_expert_per_layer_bytes": one_route_bytes,
            "routed_expert_reads_per_token": (
                int(layout_summary["moe_layer_count"]) * effective_top_k
            ),
        },
        "flash_moe_reference": reference,
        "comparison": {
            "glm_to_flash_moe_bytes_per_token_ratio": ratio,
            "glm_io_floor_seconds_ratio": (
                float(glm_floor["io_floor_seconds_per_token"])
                / float(reference["io_floor_seconds_per_token"])
            ),
            "flash_moe_reported_tokens_per_second": reference_reported_tps,
            "glm_tokens_per_second_at_flash_moe_reported_efficiency": same_efficiency_tps,
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a static routed-expert I/O envelope for GLM and compare it "
            "with the flash-moe Qwen reference without opening model weights."
        )
    )
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--expert-layout", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--io-gib-per-second", type=float)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_efficiency_envelope(
            args.prepared_dir,
            config_path=args.config,
            expert_layout_path=args.expert_layout,
            top_k=args.top_k,
            io_gib_per_second=args.io_gib_per_second,
        )
    except EfficiencyEnvelopeError as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        io = payload["routed_expert_io"]
        comp = payload["comparison"]
        print(f"GLM routed expert bytes/token: {io['gib_per_token']:.3f} GiB")
        print(
            "GLM I/O floor: "
            f"{io['io_floor_seconds_per_token']:.3f} s/token "
            f"({io['io_floor_tokens_per_second']:.3f} tok/s)"
        )
        print(
            "GLM/flash-moe routed bytes ratio: "
            f"{comp['glm_to_flash_moe_bytes_per_token_ratio']:.2f}x"
        )
        print(
            "GLM tok/s at flash-moe reported efficiency: "
            f"{comp['glm_tokens_per_second_at_flash_moe_reported_efficiency']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
