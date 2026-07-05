#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layer_moe_smoke import write_fixture


Mutator = Callable[[dict[str, object]], None]


def _run_reject_case(
    *,
    runner: Path,
    root: Path,
    name: str,
    mutator: Mutator,
    expected_error: str,
) -> None:
    case_root = root / name
    case_root.mkdir(parents=True, exist_ok=True)
    write_fixture(case_root)
    layout_path = case_root / "experts" / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    mutator(layout)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    cmd = [
        str(runner),
        "--layout",
        str(layout_path),
        "--layer",
        "1",
        "--run-moe",
        "--experts",
        "0",
        "--weights",
        "1",
        "--input-f32",
        str(case_root / "input.f32"),
        "--output-f32",
        str(case_root / "out.f32"),
        "--max-slot-mib",
        "1",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise SystemExit(f"{name}: runner accepted invalid expert layout")
    if expected_error not in combined:
        raise SystemExit(
            f"{name}: expected error {expected_error!r}; got:\n{combined}"
        )
    print(f"  rejected {name}: {expected_error}")


def _swap_component_order(layout: dict[str, object]) -> None:
    order = layout["component_order"]
    assert isinstance(order, list)
    order[1], order[2] = order[2], order[1]


def _swap_scale_bias_offsets(layout: dict[str, object]) -> None:
    layers = layout["layers"]
    assert isinstance(layers, list)
    layer = layers[0]
    assert isinstance(layer, dict)
    components = layer["components"]
    assert isinstance(components, list)
    components[1]["offset"], components[2]["offset"] = (
        components[2]["offset"],
        components[1]["offset"],
    )


def _change_weight_dtype(layout: dict[str, object]) -> None:
    layers = layout["layers"]
    assert isinstance(layers, list)
    layer = layers[0]
    assert isinstance(layer, dict)
    components = layer["components"]
    assert isinstance(components, list)
    components[0]["dtype"] = "BF16"


def _escape_layer_file(layout: dict[str, object]) -> None:
    layers = layout["layers"]
    assert isinstance(layers, list)
    layer = layers[0]
    assert isinstance(layer, dict)
    layer["layer_file"] = "../layer_001.bin"


def _use_boolean_layer_id(layout: dict[str, object]) -> None:
    layers = layout["layers"]
    assert isinstance(layers, list)
    layer = layers[0]
    assert isinstance(layer, dict)
    layer["layer"] = True


def _use_boolean_slot_bytes(layout: dict[str, object]) -> None:
    layers = layout["layers"]
    assert isinstance(layers, list)
    layer = layers[0]
    assert isinstance(layer, dict)
    layer["expert_slot_bytes"] = True


def _use_boolean_component_offset(layout: dict[str, object]) -> None:
    layers = layout["layers"]
    assert isinstance(layers, list)
    layer = layers[0]
    assert isinstance(layer, dict)
    components = layer["components"]
    assert isinstance(components, list)
    components[0]["offset"] = False


def _use_boolean_component_shape(layout: dict[str, object]) -> None:
    layers = layout["layers"]
    assert isinstance(layers, list)
    layer = layers[0]
    assert isinstance(layer, dict)
    components = layer["components"]
    assert isinstance(components, list)
    shape = components[0]["shape"]
    assert isinstance(shape, list)
    shape[0] = True


def _use_boolean_group_size(layout: dict[str, object]) -> None:
    layout["group_size"] = True


def _inflate_num_experts(layout: dict[str, object]) -> None:
    layers = layout["layers"]
    assert isinstance(layers, list)
    layer = layers[0]
    assert isinstance(layer, dict)
    layer["num_experts"] = 3
    layout["num_experts"] = 3


def _mark_mxfp4_quantization(layout: dict[str, object]) -> None:
    layout["quantization"] = "mlx-mxfp4"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify runner-side affine-int4 expert layout contract failures.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("largerlm-runner"),
        help="Path to the compiled Metal runner.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Fixture directory. Defaults to a new /private/tmp directory.",
    )
    args = parser.parse_args()

    root = args.root or Path(
        tempfile.mkdtemp(prefix="largerlm-expert-layout-contract-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    print(f"fixture: {root}")
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="component-order",
        mutator=_swap_component_order,
        expected_error="component_order does not match affine-int4 slot order",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="component-offset",
        mutator=_swap_scale_bias_offsets,
        expected_error="component gate_proj.scales span does not match",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="component-dtype",
        mutator=_change_weight_dtype,
        expected_error="component gate_proj.weight dtype does not match",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="layer-file-path",
        mutator=_escape_layer_file,
        expected_error="layer_file must be a relative file name",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="layer-id-type",
        mutator=_use_boolean_layer_id,
        expected_error="expert layout layer id must be an integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="slot-bytes-type",
        mutator=_use_boolean_slot_bytes,
        expected_error="expert layout layer expert_slot_bytes must be a positive integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="component-offset-type",
        mutator=_use_boolean_component_offset,
        expected_error="expert layout component offset must be an integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="component-shape-type",
        mutator=_use_boolean_component_shape,
        expected_error="expert layout component shape dimension must be an integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="group-size-type",
        mutator=_use_boolean_group_size,
        expected_error="expert layout group_size must be an integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="layer-file-size",
        mutator=_inflate_num_experts,
        expected_error="expert layer file size",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="mxfp4-component-order",
        mutator=_mark_mxfp4_quantization,
        expected_error="component_order does not match mlx-mxfp4 slot order",
    )
    print("  contract smoke:     ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
