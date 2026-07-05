#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.latent_value_collapse import (  # noqa: E402
    LatentValueCollapseError,
    build_latent_value_collapse_plan,
)


def _format_gib(value: int) -> str:
    return f"{value / 1024**3:.3f} GiB"


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}x"


def _parse_layers(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    layers: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise argparse.ArgumentTypeError("layer ranges must be ascending")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    return tuple(sorted(layers))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan exact GLM latent-value attention-output collapse beyond "
            "the context=1 special case."
        )
    )
    parser.add_argument("prepared_dir", type=Path)
    parser.add_argument("--dtype", choices=("BF16", "F32"), default="BF16")
    parser.add_argument("--layers", default=None, help='optional layer spec, e.g. "0,3-5"')
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        layers = _parse_layers(args.layers)
        plan = build_latent_value_collapse_plan(
            args.prepared_dir,
            dtype=args.dtype,
            layers=layers,
        )
    except (LatentValueCollapseError, ValueError, argparse.ArgumentTypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report = plan.to_report()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("GLM latent-value collapse plan")
    print(f"  prepared:                 {plan.prepared_dir}")
    print(f"  dtype:                    {plan.dtype}")
    print(f"  layers:                   {plan.layer_count}")
    print(
        "  dims:                     "
        f"hidden={plan.hidden_dim} heads={plan.num_heads} "
        f"v_head={plan.v_head_dim} kv_lora={plan.kv_lora_dim}"
    )
    print(
        "  current o_proj/token:     "
        f"{_format_gib(plan.current_o_proj_bytes_per_token)}"
    )
    print(
        "  context1 cache:           "
        f"{_format_gib(plan.context1_cache_bytes)} "
        f"({_format_ratio(plan.context1_cache_read_ratio)})"
    )
    print(
        "  exact per-head cache:     "
        f"{_format_gib(plan.exact_per_head_cache_bytes)} "
        f"({_format_ratio(plan.exact_per_head_cache_read_ratio)})"
    )
    print(
        "  exact int4 floor:         "
        f"{_format_gib(plan.exact_per_head_int4_floor_bytes)} "
        f"({_format_ratio(plan.exact_per_head_int4_floor_read_ratio)})"
    )
    print(
        "  runtime FMA ratios:       "
        f"context1={_format_ratio(plan.context1_runtime_fma_ratio)} "
        f"per-head={_format_ratio(plan.exact_per_head_runtime_fma_ratio)}"
    )
    print(
        "  break-even head groups:   "
        f"{plan.break_even_shared_head_groups} "
        f"(actual independent heads {plan.independent_attention_head_groups})"
    )
    print(f"  recommended:              {plan.exact_all_context_cache_recommended}")
    print(f"  decision:                 {plan.decision}")
    if args.output is not None:
        print(f"  report json:              {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
