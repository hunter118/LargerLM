#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.prefill_execute import (
    PREFILL_ROUTER_GATE_TENSOR_SUFFIX,
    _find_layer_matrix,
    _load_json,
    _run_router_json_batch,
    _tensor_shape2,
    _write_router_json_batch_from_logits,
    run_resident_batch_linear,
)


def _write_input(path: Path, *, batch_tokens: int, hidden_dim: int, scale: float) -> None:
    values: list[float] = []
    for token in range(batch_tokens):
        for dim in range(hidden_dim):
            x = ((token + 1) * 0.013) + ((dim % 257) * 0.00037)
            values.append(scale * (math.sin(x) + 0.5 * math.cos(0.37 * x)))
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def _read_router(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_logits(lhs: list[float], rhs: list[float]) -> dict[str, float]:
    if len(lhs) != len(rhs):
        return {"count": float(min(len(lhs), len(rhs))), "max_abs": float("inf")}
    max_abs = 0.0
    sum_abs = 0.0
    for a, b in zip(lhs, rhs):
        diff = abs(float(a) - float(b))
        max_abs = max(max_abs, diff)
        sum_abs += diff
    return {
        "count": float(len(lhs)),
        "max_abs": max_abs,
        "mean_abs": sum_abs / len(lhs) if lhs else 0.0,
    }


def _compare_dirs(
    lhs_dir: Path,
    rhs_dir: Path,
    *,
    batch_tokens: int,
    weight_atol: float,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    max_abs = 0.0
    mean_abs_sum = 0.0
    max_weight_abs = 0.0
    for token in range(batch_tokens):
        lhs = _read_router(lhs_dir / f"token_{token:06d}.router.json")
        rhs = _read_router(rhs_dir / f"token_{token:06d}.router.json")
        logits = _compare_logits(lhs.get("logits") or [], rhs.get("logits") or [])
        max_abs = max(max_abs, logits["max_abs"])
        mean_abs_sum += logits.get("mean_abs", 0.0)
        lhs_weights = tuple(float(value) for value in (lhs.get("weights") or ()))
        rhs_weights = tuple(float(value) for value in (rhs.get("weights") or ()))
        weight_abs = max(
            (abs(a - b) for a, b in zip(lhs_weights, rhs_weights)),
            default=0.0,
        )
        if len(lhs_weights) != len(rhs_weights):
            weight_abs = float("inf")
        max_weight_abs = max(max_weight_abs, weight_abs)
        if lhs.get("experts") != rhs.get("experts") or weight_abs > weight_atol:
            mismatches.append(
                {
                    "token": token,
                    "lhs_experts": lhs.get("experts"),
                    "rhs_experts": rhs.get("experts"),
                    "lhs_weights": lhs.get("weights"),
                    "rhs_weights": rhs.get("weights"),
                    "logits": logits,
                }
            )
    return {
        "match": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:8],
        "logits_max_abs": max_abs,
        "logits_mean_abs": mean_abs_sum / batch_tokens if batch_tokens else 0.0,
        "weights_max_abs": max_weight_abs,
        "weight_atol": weight_atol,
    }


def _resident_linear_router(
    *,
    runner: Path,
    resident_layout: Path,
    layer: int,
    input_f32: Path,
    output_dir: Path,
    batch_tokens: int,
    hidden_dim: int,
    num_experts: int,
    backend: str,
    top_k: int,
    router_score: str,
    routed_scaling_factor: float | None,
    norm_topk_prob: bool,
    no_norm_topk_prob: bool,
    router_n_group: int | None,
    router_topk_group: int | None,
    ignore_router_bias: bool,
    max_resident_matrix_mib: int,
    max_router_mib: int,
    max_runner_scratch_mib: int,
    mpsgraph_min_batch_tokens: int,
    mpsgraph_min_matrix_dim: int,
) -> dict[str, Any]:
    logits_path = output_dir / "router_logits.f32"
    router_json_dir = output_dir / "router_json"
    result = run_resident_batch_linear(
        runner_path=runner,
        resident_layout_path=resident_layout,
        layer=layer,
        tensor_suffix=PREFILL_ROUTER_GATE_TENSOR_SUFFIX,
        input_f32_path=input_f32,
        output_f32_path=logits_path,
        batch_tokens=batch_tokens,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=backend,
        prefill_mpsgraph_min_batch_tokens=mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=mpsgraph_min_matrix_dim,
        echo_runner_output=False,
    )
    layout = _load_json(resident_layout)
    router_margin_summary = _write_router_json_batch_from_logits(
        resident_layout_path=resident_layout,
        layout=layout,
        layer=layer,
        logits_path=result.output_path,
        router_json_dir=router_json_dir,
        batch_tokens=batch_tokens,
        num_experts=num_experts,
        top_k=top_k,
        router_score=router_score,
        routed_scaling_factor=routed_scaling_factor,
        norm_topk_prob=norm_topk_prob,
        no_norm_topk_prob=no_norm_topk_prob,
        router_n_group=router_n_group,
        router_topk_group=router_topk_group,
        ignore_router_bias=ignore_router_bias,
        max_router_mib=max_router_mib,
    )
    if result.in_dim != hidden_dim or result.out_dim != num_experts:
        raise SystemExit(
            f"unexpected router shape {result.out_dim}x{result.in_dim}; "
            f"expected {num_experts}x{hidden_dim}"
        )
    return {
        "backend": result.backend,
        "elapsed_seconds": result.elapsed_seconds,
        "router_json_dir": str(router_json_dir),
        "router_margin_summary": router_margin_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared",
        default="artifacts/glm-5.2-mxfp4/largerlm-prepared",
    )
    parser.add_argument("--runner", default="metal/largerlm-runner")
    parser.add_argument("--layer", type=int, default=19)
    parser.add_argument("--batch-tokens", type=int, default=17)
    parser.add_argument("--input-scale", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--router-score", default="sigmoid", choices=("sigmoid", "softmax", "raw"))
    parser.add_argument("--routed-scaling-factor", type=float, default=1.0)
    parser.add_argument("--norm-topk-prob", action="store_true")
    parser.add_argument("--no-norm-topk-prob", action="store_true")
    parser.add_argument("--router-n-group", type=int, default=8)
    parser.add_argument("--router-topk-group", type=int, default=4)
    parser.add_argument("--ignore-router-bias", action="store_true")
    parser.add_argument("--max-resident-matrix-mib", type=int, default=64)
    parser.add_argument("--max-router-mib", type=int, default=64)
    parser.add_argument("--max-runner-scratch-mib", type=int, default=512)
    parser.add_argument("--mpsgraph-min-batch-tokens", type=int, default=16)
    parser.add_argument("--mpsgraph-min-matrix-dim", type=int, default=32)
    parser.add_argument("--weight-atol", type=float, default=1e-5)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    prepared = Path(args.prepared)
    runner = Path(args.runner)
    resident_layout = prepared / "resident" / "layout.json"
    layout = _load_json(resident_layout)
    router = _find_layer_matrix(
        layout,
        layer=args.layer,
        suffix=PREFILL_ROUTER_GATE_TENSOR_SUFFIX,
    )
    num_experts, hidden_dim = _tensor_shape2(router, "router gate")

    with tempfile.TemporaryDirectory(prefix="largerlm-router-gate-", dir="/private/tmp") as tmp:
        root = Path(tmp)
        input_f32 = root / "input.f32"
        _write_input(
            input_f32,
            batch_tokens=args.batch_tokens,
            hidden_dim=hidden_dim,
            scale=args.input_scale,
        )

        fallback_dir = root / "fallback_router_batch"
        (
            command_count,
            first_command,
            fallback_gate,
            fallback_margin_summary,
        ) = _run_router_json_batch(
            runner_path=runner,
            resident_layout_path=resident_layout,
            layer=args.layer,
            input_f32_path=input_f32,
            output_dir=fallback_dir,
            router_json_dir=fallback_dir / "router_json",
            batch_tokens=args.batch_tokens,
            hidden_dim=hidden_dim,
            top_k=args.top_k,
            router_score=args.router_score,
            routed_scaling_factor=args.routed_scaling_factor,
            norm_topk_prob=args.norm_topk_prob,
            no_norm_topk_prob=args.no_norm_topk_prob,
            router_n_group=args.router_n_group,
            router_topk_group=args.router_topk_group,
            ignore_router_bias=args.ignore_router_bias,
            max_resident_matrix_mib=args.max_resident_matrix_mib,
            max_router_mib=args.max_router_mib,
            max_runner_scratch_mib=args.max_runner_scratch_mib,
            prefill_linear_backend="custom-metal",
            prefill_mpsgraph_min_batch_tokens=args.mpsgraph_min_batch_tokens,
            prefill_mpsgraph_min_matrix_dim=args.mpsgraph_min_matrix_dim,
            keep_token_files=False,
            echo_runner_output=False,
        )
        if fallback_gate is not None:
            raise SystemExit("custom-metal fallback unexpectedly returned router gate result")

        custom = _resident_linear_router(
            runner=runner,
            resident_layout=resident_layout,
            layer=args.layer,
            input_f32=input_f32,
            output_dir=root / "custom_resident_linear",
            batch_tokens=args.batch_tokens,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            backend="custom-metal",
            top_k=args.top_k,
            router_score=args.router_score,
            routed_scaling_factor=args.routed_scaling_factor,
            norm_topk_prob=args.norm_topk_prob,
            no_norm_topk_prob=args.no_norm_topk_prob,
            router_n_group=args.router_n_group,
            router_topk_group=args.router_topk_group,
            ignore_router_bias=args.ignore_router_bias,
            max_resident_matrix_mib=args.max_resident_matrix_mib,
            max_router_mib=args.max_router_mib,
            max_runner_scratch_mib=args.max_runner_scratch_mib,
            mpsgraph_min_batch_tokens=args.mpsgraph_min_batch_tokens,
            mpsgraph_min_matrix_dim=args.mpsgraph_min_matrix_dim,
        )
        mpsgraph = _resident_linear_router(
            runner=runner,
            resident_layout=resident_layout,
            layer=args.layer,
            input_f32=input_f32,
            output_dir=root / "mpsgraph_resident_linear",
            batch_tokens=args.batch_tokens,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            backend="mpsgraph-f32",
            top_k=args.top_k,
            router_score=args.router_score,
            routed_scaling_factor=args.routed_scaling_factor,
            norm_topk_prob=args.norm_topk_prob,
            no_norm_topk_prob=args.no_norm_topk_prob,
            router_n_group=args.router_n_group,
            router_topk_group=args.router_topk_group,
            ignore_router_bias=args.ignore_router_bias,
            max_resident_matrix_mib=args.max_resident_matrix_mib,
            max_router_mib=args.max_router_mib,
            max_runner_scratch_mib=args.max_runner_scratch_mib,
            mpsgraph_min_batch_tokens=args.mpsgraph_min_batch_tokens,
            mpsgraph_min_matrix_dim=args.mpsgraph_min_matrix_dim,
        )

        fallback_router_dir = fallback_dir / "router_json"
        custom_router_dir = Path(custom["router_json_dir"])
        mpsgraph_router_dir = Path(mpsgraph["router_json_dir"])
        payload = {
            "schema": "largerlm.router_gate_consistency.v1",
            "layer": args.layer,
            "batch_tokens": args.batch_tokens,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "top_k": args.top_k,
            "router_score": args.router_score,
            "routed_scaling_factor": args.routed_scaling_factor,
            "router_n_group": args.router_n_group,
            "router_topk_group": args.router_topk_group,
            "fallback_router_batch": {
                "command_count": command_count,
                "first_command": list(first_command),
                "router_json_dir": str(fallback_router_dir),
                "router_margin_summary": fallback_margin_summary,
            },
            "custom_resident_linear": custom,
            "mpsgraph_resident_linear": mpsgraph,
            "comparisons": {
                "fallback_vs_custom_resident": _compare_dirs(
                    fallback_router_dir,
                    custom_router_dir,
                    batch_tokens=args.batch_tokens,
                    weight_atol=args.weight_atol,
                ),
                "fallback_vs_mpsgraph_resident": _compare_dirs(
                    fallback_router_dir,
                    mpsgraph_router_dir,
                    batch_tokens=args.batch_tokens,
                    weight_atol=args.weight_atol,
                ),
                "custom_resident_vs_mpsgraph_resident": _compare_dirs(
                    custom_router_dir,
                    mpsgraph_router_dir,
                    batch_tokens=args.batch_tokens,
                    weight_atol=args.weight_atol,
                ),
            },
        }

    if args.output_json is not None:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
