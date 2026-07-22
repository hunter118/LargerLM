#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.expert_usage import (
    GIB,
    M5_MAX_128G_SAFE_PIN_BUDGET_BYTES,
    ExpertUsageError,
    build_pin_plan,
    build_usage_profile,
    format_colibri_usage,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an expert usage profile and bounded RAM pin plan."
    )
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--expert-layout", type=Path)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--pin-budget-gib", type=float)
    budget.add_argument(
        "--m5-max-128g-safe",
        action="store_true",
        help=(
            "Use 44 GiB hard-pinned experts plus a planned 36 GiB evictable "
            "cache for a 128 GiB M5 Max."
        ),
    )
    parser.add_argument("--default-layer", type=int)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--max-experts-per-layer", type=int)
    parser.add_argument("--write-profile", type=Path)
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--write-colibri-usage", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        profile = build_usage_profile(args.sources, default_layer=args.default_layer)
        if args.write_profile is not None:
            _write_json(args.write_profile, profile)
        if args.write_colibri_usage is not None:
            args.write_colibri_usage.parent.mkdir(parents=True, exist_ok=True)
            args.write_colibri_usage.write_text(
                format_colibri_usage(profile), encoding="utf-8"
            )

        plan = None
        if args.expert_layout is not None:
            if args.m5_max_128g_safe:
                max_pin_bytes = M5_MAX_128G_SAFE_PIN_BUDGET_BYTES
                target_profile = "m5-max-128g-safe"
            elif args.pin_budget_gib is not None and args.pin_budget_gib > 0:
                max_pin_bytes = int(args.pin_budget_gib * GIB)
                target_profile = None
            else:
                raise ExpertUsageError(
                    "an expert layout requires --pin-budget-gib or --m5-max-128g-safe"
                )
            plan = build_pin_plan(
                profile,
                expert_layout_path=args.expert_layout,
                max_pin_bytes=max_pin_bytes,
                min_count=args.min_count,
                max_experts_per_layer=args.max_experts_per_layer,
                target_profile=target_profile,
            )
            if args.write_plan is not None:
                _write_json(args.write_plan, plan)
        elif args.write_plan is not None:
            raise ExpertUsageError("--write-plan requires --expert-layout")

        if not args.quiet:
            print(json.dumps(plan if plan is not None else profile, indent=2))
    except ExpertUsageError as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
