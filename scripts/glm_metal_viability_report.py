#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.metal_viability import (  # noqa: E402
    MetalViabilityError,
    build_metal_viability_report,
)


def _format_gib(bytes_value: int) -> str:
    return f"{bytes_value / 1024**3:.3f} GiB"


def _format_optional_gib(bytes_value: int | None) -> str:
    if bytes_value is None:
        return "n/a"
    return _format_gib(bytes_value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize whether the current GLM Metal runtime is merely runnable "
            "or plausibly usable-speed."
        )
    )
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--smoke-result", type=Path, default=None)
    parser.add_argument("--ssd-read-gib-s", type=float, default=None)
    parser.add_argument(
        "--context1-collapse-plan",
        type=Path,
        default=None,
        help=(
            "optional context=1 o_proj*B_v collapse plan; defaults to the "
            "latest report in the prepared directory when present"
        ),
    )
    parser.add_argument(
        "--decode-telemetry",
        type=Path,
        default=None,
        help=(
            "optional Metal generation result or decode_telemetry_report JSON "
            "used to project collapsed-cache impact"
        ),
    )
    parser.add_argument(
        "--target-tok-s",
        type=float,
        default=None,
        help=(
            "optional product throughput target; the report compares the best "
            "observed/projected throughput against this gate"
        ),
    )
    parser.add_argument(
        "--require-below-target",
        action="store_true",
        help=(
            "exit non-zero unless --target-tok-s is set and the best "
            "observed/projected throughput remains below that target"
        ),
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        smoke_result = args.smoke_result
        if smoke_result is None and args.decode_telemetry is not None:
            smoke_result = args.decode_telemetry
        report = build_metal_viability_report(
            args.prepared_dir,
            top_k=args.top_k,
            smoke_result_path=smoke_result,
            ssd_read_gib_per_second=args.ssd_read_gib_s,
            context1_collapse_plan_path=args.context1_collapse_plan,
            decode_telemetry_path=args.decode_telemetry,
            target_tokens_per_second=args.target_tok_s,
        )
    except MetalViabilityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.require_below_target:
        if args.target_tok_s is None:
            print(
                "ERROR: --require-below-target requires --target-tok-s",
                file=sys.stderr,
            )
            return 2
        if report.target_met is not False:
            status = "met" if report.target_met is True else "unknown"
            print(
                "ERROR: throughput target is not below threshold "
                f"(status={status}, target={args.target_tok_s})",
                file=sys.stderr,
            )
            return 1

    if args.json:
        print(json.dumps(report.to_json_dict(), indent=2, sort_keys=True))
        return 0

    print("GLM Metal viability")
    print(f"  prepared:                 {report.prepared_dir}")
    print(f"  expert layers:            {report.expert_layer_count}")
    print(f"  top-k:                    {report.top_k}")
    print(
        "  expert slot bytes:        "
        f"{_format_gib(report.expert_slot_bytes_min)}.."
        f"{_format_gib(report.expert_slot_bytes_max)}"
    )
    print(
        "  routed read/token:        "
        f"{_format_gib(report.routed_expert_read_bytes_per_token)}"
    )
    print(
        "  GLM/Qwen read ratio:      "
        f"{report.glm_to_qwen_flash_moe_read_ratio:.2f}x"
    )
    if report.ssd_read_gib_per_second is not None:
        print(f"  SSD read model:           {report.ssd_read_gib_per_second:.3f} GiB/s")
    if report.routed_read_lower_bound_seconds_per_token is not None:
        print(
            "  routed-read lower bound: "
            f"{report.routed_read_lower_bound_seconds_per_token:.3f} s/token"
        )
    if report.observed_decode_seconds_per_token is not None:
        print(
            "  observed decode:         "
            f"{report.observed_decode_seconds_per_token:.3f} s/token"
        )
    if report.observed_tokens_per_second is not None:
        print(f"  observed throughput:      {report.observed_tokens_per_second:.3f} tok/s")
    if report.observed_generated_token_ids:
        print(
            "  observed token ids:      "
            + ",".join(str(item) for item in report.observed_generated_token_ids)
        )
    if report.context1_cache_bytes is not None:
        print(
            "  context1 cache:          "
            f"{_format_optional_gib(report.context1_cache_bytes)} "
            f"({report.context1_layers_supported} layers)"
        )
        if report.context1_current_o_proj_bytes_per_token is not None:
            print(
                "  current o_proj/token:    "
                f"{_format_gib(report.context1_current_o_proj_bytes_per_token)}"
            )
        if report.context1_cache_read_ratio is not None:
            print(
                "  context1 read ratio:     "
                f"{report.context1_cache_read_ratio:.3f}x"
            )
        if report.context1_build_fma_per_layer is not None:
            print(
                "  context1 layer build:    "
                f"{report.context1_build_fma_per_layer / 1.0e9:.3f} GFMA"
            )
    if report.observed_decode_token_count is not None:
        print(f"  telemetry tokens:        {report.observed_decode_token_count}")
        if report.observed_attn_output_bytes_per_token is not None:
            print(
                "  observed o_proj/token:   "
                f"{_format_gib(int(report.observed_attn_output_bytes_per_token))}"
            )
        if report.observed_attn_output_read_seconds_per_token is not None:
            print(
                "  observed o_proj read:    "
                f"{report.observed_attn_output_read_seconds_per_token:.3f} s/token"
            )
        if report.observed_attn_output_projection_seconds_per_token is not None:
            print(
                "  observed o_proj kernel:  "
                f"{report.observed_attn_output_projection_seconds_per_token:.3f} s/token"
            )
    if report.projected_context1_only_tokens_per_second is not None:
        print(
            "  projected context1-only: "
            f"{report.projected_context1_only_tokens_per_second:.3f} tok/s "
            f"({report.projected_context1_only_speedup:.2f}x)"
        )
    if report.exact_all_context_per_head_cache_bytes is not None:
        print(
            "  exact per-head cache:    "
            f"{_format_gib(report.exact_all_context_per_head_cache_bytes)} "
            f"({report.exact_all_context_per_head_cache_read_ratio:.2f}x read)"
        )
    if report.exact_all_context_per_head_int4_floor_bytes is not None:
        print(
            "  exact int4 floor:        "
            f"{_format_gib(report.exact_all_context_per_head_int4_floor_bytes)} "
            f"({report.exact_all_context_per_head_int4_floor_read_ratio:.2f}x read)"
        )
    if report.exact_all_context_break_even_shared_head_groups is not None:
        print(
            "  break-even head groups:  "
            f"{report.exact_all_context_break_even_shared_head_groups} "
            f"(actual {report.exact_all_context_independent_attention_head_groups})"
        )
    if report.projected_exact_all_context_per_head_tokens_per_second is not None:
        print(
            "  projected per-head exact:"
            f" {report.projected_exact_all_context_per_head_tokens_per_second:.3f} tok/s "
            f"({report.projected_exact_all_context_per_head_speedup:.2f}x)"
        )
    if report.efficiency_rewrite_decision is not None:
        print(f"  efficiency decision:     {report.efficiency_rewrite_decision}")
    if report.target_tokens_per_second is not None:
        print(f"  target throughput:       {report.target_tokens_per_second:.3f} tok/s")
        if report.target_reference_tokens_per_second is not None:
            print(
                "  best observed/projected:"
                f" {report.target_reference_tokens_per_second:.3f} tok/s"
            )
        if report.target_reference_source is not None:
            print(f"  target basis:            {report.target_reference_source}")
        if report.target_gap_multiplier is not None:
            print(f"  target gap:              {report.target_gap_multiplier:.2f}x")
        if report.target_decision is not None:
            print(f"  target decision:         {report.target_decision}")
    print(f"  decision:                 {report.decision}")
    print(f"  next runtime step:        {report.next_runtime_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
