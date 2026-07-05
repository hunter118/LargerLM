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
    layout_path = case_root / "resident" / "layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    mutator(layout)
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    cmd = [
        str(runner),
        "--resident-layout",
        str(layout_path),
        "--layer",
        "1",
        "--run-router",
        "--input-f32",
        str(case_root / "input.f32"),
        "--top-k",
        "1",
        "--router-score",
        "raw",
        "--max-router-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise SystemExit(f"{name}: runner accepted invalid resident layout")
    if expected_error not in combined:
        raise SystemExit(
            f"{name}: expected error {expected_error!r}; got:\n{combined}"
        )
    print(f"  rejected {name}: {expected_error}")


def _escape_weight_file(layout: dict[str, object]) -> None:
    layout["weight_file"] = "../resident.bin"


def _use_boolean_total_bytes(layout: dict[str, object]) -> None:
    layout["total_bytes"] = True


def _inflate_total_bytes(layout: dict[str, object]) -> None:
    layout["total_bytes"] = int(layout["total_bytes"]) + 4


def _use_boolean_tensor_offset(layout: dict[str, object]) -> None:
    tensor = list(layout["tensors"])[0]
    assert isinstance(tensor, dict)
    tensor["offset"] = False


def _shrink_tensor_size(layout: dict[str, object]) -> None:
    tensor = list(layout["tensors"])[0]
    assert isinstance(tensor, dict)
    tensor["size"] = int(tensor["size"]) - 4


def _move_tensor_past_total(layout: dict[str, object]) -> None:
    tensor = list(layout["tensors"])[0]
    assert isinstance(tensor, dict)
    tensor["offset"] = int(layout["total_bytes"]) - 4


def _duplicate_tensor_span(layout: dict[str, object]) -> None:
    tensors = layout["tensors"]
    assert isinstance(tensors, list)
    tensors.append(dict(tensors[0]))


def _use_unsupported_tensor_dtype(layout: dict[str, object]) -> None:
    tensor = list(layout["tensors"])[0]
    assert isinstance(tensor, dict)
    tensor["dtype"] = "I32"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify runner-side resident layout backing-file contract failures.",
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
        tempfile.mkdtemp(prefix="largerlm-resident-layout-contract-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    print(f"fixture: {root}")
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="weight-file-path",
        mutator=_escape_weight_file,
        expected_error="weight_file must be a relative file name",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="total-bytes-type",
        mutator=_use_boolean_total_bytes,
        expected_error="resident layout total_bytes must be a positive integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="weight-file-size",
        mutator=_inflate_total_bytes,
        expected_error="resident weight file size",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="tensor-offset-type",
        mutator=_use_boolean_tensor_offset,
        expected_error="resident tensor offset must be an integer",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="tensor-size-mismatch",
        mutator=_shrink_tensor_size,
        expected_error="resident tensor model.layers.1.mlp.gate.weight size mismatch",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="tensor-span-exceeds-total",
        mutator=_move_tensor_past_total,
        expected_error="resident tensor model.layers.1.mlp.gate.weight exceeds layout total_bytes",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="tensor-span-overlap",
        mutator=_duplicate_tensor_span,
        expected_error="resident tensor byte spans overlap",
    )
    _run_reject_case(
        runner=args.runner,
        root=root,
        name="tensor-dtype",
        mutator=_use_unsupported_tensor_dtype,
        expected_error="unsupported resident tensor dtype I32",
    )
    print("  contract smoke:     ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
