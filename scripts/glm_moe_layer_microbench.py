#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from largerlm.expert_io import stage_batch_experts
from largerlm.staged_moe import run_staged_routed_moe_batch


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_layer(layout: dict[str, Any], layer_id: int) -> dict[str, Any]:
    for layer in layout.get("layers", []):
        if isinstance(layer, dict) and layer.get("layer") == layer_id:
            return layer
    raise SystemExit(f"layer {layer_id} not found in {layout}")


def _component_shape(layer: dict[str, Any], name: str) -> tuple[int, int]:
    for component in layer.get("components", []):
        if isinstance(component, dict) and component.get("name") == name:
            shape = component.get("shape")
            if (
                isinstance(shape, list)
                and len(shape) == 2
                and type(shape[0]) is int
                and type(shape[1]) is int
            ):
                return int(shape[0]), int(shape[1])
    raise SystemExit(f"component {name} not found")


def _parse_experts(value: str) -> tuple[int, ...]:
    experts = tuple(int(item) for item in value.split(",") if item.strip())
    if not experts:
        raise argparse.ArgumentTypeError("expert list must not be empty")
    if any(expert < 0 for expert in experts):
        raise argparse.ArgumentTypeError("expert ids must be non-negative")
    return experts


def _write_router_jsons(
    *,
    router_json_dir: Path,
    batch_tokens: int,
    experts: tuple[int, ...],
) -> None:
    router_json_dir.mkdir(parents=True, exist_ok=True)
    weight = 1.0 / float(len(experts))
    payload = {"experts": list(experts), "weights": [weight] * len(experts)}
    text = json.dumps(payload, separators=(",", ":"))
    for token in range(batch_tokens):
        (router_json_dir / f"token_{token:06d}.router.json").write_text(
            text,
            encoding="utf-8",
        )


def _write_zero_input(path: Path, *, batch_tokens: int, hidden_dim: int) -> None:
    row = struct.pack("<" + "f" * hidden_dim, *([0.0] * hidden_dim))
    with path.open("wb") as handle:
        for _ in range(batch_tokens):
            handle.write(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one real GLM staged routed-MoE layer as a bounded microbench.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--runner", type=Path, default=Path("metal/largerlm-runner"))
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument(
        "--experts",
        type=_parse_experts,
        default="11,79,92,103,154,212,236,254",
        help="comma-separated original expert ids",
    )
    parser.add_argument("--max-stage-mib", type=float, default=512.0)
    parser.add_argument("--max-compact-stage-mib", type=float, default=512.0)
    parser.add_argument("--copy-chunk-mib", type=float, default=8.0)
    parser.add_argument("--max-slot-mib", type=int, default=256)
    parser.add_argument("--max-runner-scratch-mib", type=int, default=4096)
    parser.add_argument("--moe-token-block", default="auto")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--write-result", type=Path)
    args = parser.parse_args()

    if args.batch_tokens <= 0:
        raise SystemExit("--batch-tokens must be positive")

    expert_layout = args.prepared_dir / "experts" / "layout.json"
    layout = _load_json(expert_layout)
    layer = _find_layer(layout, args.layer)
    hidden_dim, _ = _component_shape(layer, "down_proj.weight")
    num_experts = int(layer.get("num_experts", 0))
    for expert in args.experts:
        if expert >= num_experts:
            raise SystemExit(f"expert {expert} out of range for {num_experts} experts")

    work_dir = Path(tempfile.mkdtemp(prefix="largerlm-glm-moe-layer-"))
    cleaned = False
    try:
        router_json_dir = work_dir / "router_json"
        input_f32 = work_dir / "input.f32"
        stage_file = work_dir / "experts.stage.bin"
        stage_manifest = work_dir / "experts.stage.manifest.json"
        output_dir = work_dir / "staged_moe"
        output_f32 = work_dir / "output.f32"

        _write_router_jsons(
            router_json_dir=router_json_dir,
            batch_tokens=args.batch_tokens,
            experts=args.experts,
        )
        _write_zero_input(
            input_f32,
            batch_tokens=args.batch_tokens,
            hidden_dim=hidden_dim,
        )

        stage_started = time.perf_counter()
        stage = stage_batch_experts(
            expert_layout,
            layer=args.layer,
            router_json_dir=router_json_dir,
            stage_file_path=stage_file,
            manifest_path=stage_manifest,
            max_stage_mib=args.max_stage_mib,
            copy_chunk_mib=args.copy_chunk_mib,
        )
        stage_elapsed = time.perf_counter() - stage_started

        run_started = time.perf_counter()
        moe = run_staged_routed_moe_batch(
            runner_path=args.runner,
            stage_manifest_path=stage_manifest,
            input_f32_path=input_f32,
            output_f32_path=output_f32,
            output_dir=output_dir,
            max_compact_stage_mib=args.max_compact_stage_mib,
            copy_chunk_mib=args.copy_chunk_mib,
            max_slot_mib=args.max_slot_mib,
            max_runner_scratch_mib=args.max_runner_scratch_mib,
            moe_token_block=args.moe_token_block,
            static_capacity_per_expert=args.batch_tokens,
            write_static_capacity_json=False,
            echo_runner_output=False,
        )
        run_elapsed = time.perf_counter() - run_started

        payload = {
            "prepared_dir": args.prepared_dir,
            "runner": args.runner,
            "work_dir": work_dir,
            "layer": args.layer,
            "batch_tokens": args.batch_tokens,
            "experts": args.experts,
            "hidden_dim": hidden_dim,
            "stage_elapsed_seconds": stage_elapsed,
            "run_elapsed_seconds": run_elapsed,
            "total_elapsed_seconds": stage_elapsed + run_elapsed,
            "stage_staged_bytes": stage.staged_bytes,
            "stage_copy_elapsed_seconds": stage.copy_elapsed_seconds,
            "stage_copy_throughput_gib_per_second": (
                stage.copy_throughput_gib_per_second
            ),
            "stage_raw_range_count": stage.io_summary.raw_range_count,
            "stage_coalesced_range_count": stage.io_summary.coalesced_range_count,
            "moe_effective_token_block": moe.effective_moe_token_block,
            "moe_max_expert_tokens": moe.moe_max_expert_tokens,
            "moe_batch_buffer_bytes": moe.moe_batch_buffer_bytes,
            "moe_estimated_peak_bytes": moe.moe_estimated_peak_bytes,
            "moe_output_accumulator": moe.moe_output_accumulator,
            "moe_output_accumulator_bytes": moe.moe_output_accumulator_bytes,
            "moe_timing_elapsed_seconds": {
                "sort": moe.moe_timing_sort_seconds,
                "setup": moe.moe_timing_setup_seconds,
                "expert_read": moe.moe_timing_expert_read_seconds,
                "input_read": moe.moe_timing_input_read_seconds,
                "output_read": moe.moe_timing_output_read_seconds,
                "kernel": moe.moe_timing_kernel_seconds,
                "mxfp4_swiglu_kernel": (
                    moe.moe_timing_mxfp4_swiglu_kernel_seconds
                ),
                "mxfp4_down_add_kernel": (
                    moe.moe_timing_mxfp4_down_add_kernel_seconds
                ),
                "output_write": moe.moe_timing_output_write_seconds,
                "final_read": moe.moe_timing_final_read_seconds,
                "total": moe.moe_timing_total_seconds,
            },
            "moe_static_capacity_used_slots": moe.static_capacity_used_slots,
            "moe_static_capacity_total_slots": moe.static_capacity_total_slots,
            "moe_output_bytes": moe.output_bytes,
            "work_dir_cleaned": False,
        }
        if not args.keep_work_dir:
            shutil.rmtree(work_dir)
            cleaned = True
            payload["work_dir_cleaned"] = True
        if args.write_result is not None:
            args.write_result.parent.mkdir(parents=True, exist_ok=True)
            args.write_result.write_text(
                json.dumps(payload, default=_json_default, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, default=_json_default, indent=2, sort_keys=True))
        return 0
    finally:
        if not args.keep_work_dir and not cleaned and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
