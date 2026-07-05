#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from layer_moe_smoke import write_fixture


HEADER = struct.Struct("<8sIIIIIIII")
EXPERT = struct.Struct("<I")
SLOT = struct.Struct("<IfI")
OVERFLOW = struct.Struct("<IIIf")
INACTIVE_TOKEN = 0xFFFFFFFF


def _base_cmd(
    runner: Path,
    case_root: Path,
    *,
    routes_flag: str,
    routes_path: Path,
) -> list[str]:
    return [
        str(runner),
        "--layout",
        str(case_root / "experts" / "layout.json"),
        "--layer",
        "1",
        "--run-moe-batch",
        routes_flag,
        str(routes_path),
        "--input-f32",
        str(case_root / "batch_input.f32"),
        "--batch-tokens",
        "2",
        "--output-f32",
        str(case_root / "out.f32"),
        "--max-k",
        "2",
        "--max-slot-mib",
        "1",
        "--max-runner-scratch-mib",
        "64",
    ]


def _write_case_fixture(root: Path, name: str) -> Path:
    case_root = root / name
    case_root.mkdir(parents=True, exist_ok=True)
    write_fixture(case_root)
    case_root.joinpath("batch_input.f32").write_bytes(
        struct.pack("<16f", *([1.0] * 16))
    )
    return case_root


def _run_reject_case(cmd: list[str], *, name: str, expected_error: str) -> None:
    completed = subprocess.run(cmd, text=True, capture_output=True)
    combined = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise SystemExit(f"{name}: runner accepted invalid MoE route input")
    if "Metal device unavailable" in combined:
        raise SystemExit(
            f"{name}: route rejection happened after Metal device creation:\n{combined}"
        )
    if expected_error not in combined:
        raise SystemExit(
            f"{name}: expected error {expected_error!r}; got:\n{combined}"
        )
    print(f"  rejected {name}: {expected_error}")


def _write_routes_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _valid_routes_payload() -> dict[str, Any]:
    return {
        "batch_tokens": 2,
        "routes": [
            {"experts": [0], "weights": [1.0]},
            {"experts": [1], "weights": [1.0]},
        ],
    }


def _write_static_capacity_bin(
    path: Path,
    *,
    experts: tuple[int, ...] = (0, 1),
    capacity: int = 1,
    slots: tuple[tuple[int, float, int], ...],
    overflow: tuple[tuple[int, int, int, float], ...] = (),
) -> None:
    active_slots = sum(1 for _token, _weight, active in slots if active)
    total_assignments = active_slots + len(overflow)
    path.write_bytes(
        b"".join(
            (
                HEADER.pack(
                    b"LLMSCAP1",
                    1,
                    2,
                    len(experts),
                    capacity,
                    total_assignments,
                    active_slots,
                    len(overflow),
                    0,
                ),
                b"".join(EXPERT.pack(expert) for expert in experts),
                b"".join(SLOT.pack(token, weight, active) for token, weight, active in slots),
                b"".join(
                    OVERFLOW.pack(expert, overflow_index, token, weight)
                    for expert, overflow_index, token, weight in overflow
                ),
            )
        )
    )


def _json_cases(runner: Path, root: Path) -> None:
    cases: list[tuple[str, dict[str, Any], str]] = []

    payload = _valid_routes_payload()
    payload["batch_tokens"] = True
    cases.append(
        (
            "json-batch-tokens-bool",
            payload,
            "routes batch_tokens must be an integer",
        )
    )

    payload = _valid_routes_payload()
    payload["routes"][0]["experts"][0] = True
    cases.append(("json-expert-bool", payload, "route expert must be an integer"))

    payload = _valid_routes_payload()
    payload["routes"][0]["weights"][0] = False
    cases.append(
        (
            "json-weight-bool",
            payload,
            "route weight must be a finite number",
        )
    )

    payload = _valid_routes_payload()
    payload["routes"][0] = {"experts": [0, 0], "weights": [0.5, 0.5]}
    cases.append(
        (
            "json-duplicate-expert",
            payload,
            "route 0 contains duplicate expert 0",
        )
    )

    for name, route_payload, expected_error in cases:
        case_root = _write_case_fixture(root, name)
        routes_path = case_root / "routes.json"
        _write_routes_json(routes_path, route_payload)
        _run_reject_case(
            _base_cmd(
                runner,
                case_root,
                routes_flag="--routes-json",
                routes_path=routes_path,
            ),
            name=name,
            expected_error=expected_error,
        )


def _binary_cases(runner: Path, root: Path) -> None:
    duplicate_root = _write_case_fixture(root, "binary-duplicate-assignment")
    duplicate_path = duplicate_root / "routes.bin"
    _write_static_capacity_bin(
        duplicate_path,
        slots=((0, 1.0, 1), (1, 1.0, 1)),
        overflow=((0, 0, 0, 0.5),),
    )
    _run_reject_case(
        _base_cmd(
            runner,
            duplicate_root,
            routes_flag="--routes-bin",
            routes_path=duplicate_path,
        ),
        name="binary-duplicate-assignment",
        expected_error="duplicate MoE batch assignment for token 0 expert 0",
    )

    inf_root = _write_case_fixture(root, "binary-inf-weight")
    inf_path = inf_root / "routes.bin"
    _write_static_capacity_bin(
        inf_path,
        slots=((0, float("inf"), 1), (1, 1.0, 1)),
    )
    _run_reject_case(
        _base_cmd(
            runner,
            inf_root,
            routes_flag="--routes-bin",
            routes_path=inf_path,
        ),
        name="binary-inf-weight",
        expected_error="static capacity route weight must be finite",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify runner-side MoE batch route input contract failures.",
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
        tempfile.mkdtemp(prefix="largerlm-moe-route-contract-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    print(f"fixture: {root}")
    _json_cases(args.runner, root)
    _binary_cases(args.runner, root)
    print("  MoE route contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
