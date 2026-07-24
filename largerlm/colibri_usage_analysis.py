from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


class ColibriUsageAnalysisError(ValueError):
    """Raised when a Colibri usage file is malformed."""


def load_colibri_usage(path: Path) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ColibriUsageAnalysisError(f"cannot read {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ColibriUsageAnalysisError(
                f"{path}:{line_number} must contain layer expert count"
            )
        try:
            layer, expert, count = (int(field) for field in fields)
        except ValueError as exc:
            raise ColibriUsageAnalysisError(
                f"{path}:{line_number} contains a non-integer field"
            ) from exc
        if layer < 0 or expert < 0 or count <= 0:
            raise ColibriUsageAnalysisError(
                f"{path}:{line_number} requires layer/expert >= 0 and count > 0"
            )
        key = (layer, expert)
        counts[key] = counts.get(key, 0) + count
    if not counts:
        raise ColibriUsageAnalysisError(f"{path} has no usage rows")
    return counts


def _ranked_keys(counts: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    return sorted(counts, key=lambda key: (-counts[key], key[0], key[1]))


def _coverage(
    selected: set[tuple[int, int]],
    counts: dict[tuple[int, int], int],
) -> int:
    return sum(count for key, count in counts.items() if key in selected)


def _slots_for_fraction(
    counts: dict[tuple[int, int], int],
    fraction: float,
) -> int:
    target = sum(counts.values()) * fraction
    cumulative = 0
    for index, key in enumerate(_ranked_keys(counts), start=1):
        cumulative += counts[key]
        if cumulative >= target:
            return index
    return len(counts)


def analyze_usage_overlap(
    training: dict[tuple[int, int], int],
    evaluation: dict[tuple[int, int], int],
    *,
    slots: int,
) -> dict[str, object]:
    if slots <= 0:
        raise ColibriUsageAnalysisError("slots must be positive")
    training_ranked = _ranked_keys(training)
    evaluation_ranked = _ranked_keys(evaluation)
    training_set = set(training_ranked[:slots])
    evaluation_set = set(evaluation_ranked[:slots])
    training_total = sum(training.values())
    evaluation_total = sum(evaluation.values())
    training_covered = _coverage(training_set, training)
    heldout_covered = _coverage(training_set, evaluation)
    oracle_covered = _coverage(evaluation_set, evaluation)
    union = training_set | evaluation_set
    intersection = training_set & evaluation_set
    return {
        "schema": "largerlm.colibri_usage_overlap.v1",
        "slots": slots,
        "training": {
            "selections": training_total,
            "distinct_experts": len(training),
            "covered_selections": training_covered,
            "coverage": training_covered / training_total,
        },
        "heldout": {
            "selections": evaluation_total,
            "distinct_experts": len(evaluation),
            "training_profile_covered_selections": heldout_covered,
            "training_profile_coverage": heldout_covered / evaluation_total,
            "oracle_covered_selections": oracle_covered,
            "oracle_coverage": oracle_covered / evaluation_total,
            "oracle_slots_for_90_percent": _slots_for_fraction(evaluation, 0.90),
            "oracle_slots_for_92_percent": _slots_for_fraction(evaluation, 0.92),
            "oracle_slots_for_95_percent": _slots_for_fraction(evaluation, 0.95),
        },
        "selected_set_overlap": {
            "intersection": len(intersection),
            "union": len(union),
            "jaccard": len(intersection) / len(union) if union else 1.0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure cross-workload coverage of a Colibri hot-expert profile."
    )
    parser.add_argument("training", type=Path)
    parser.add_argument("heldout", type=Path)
    parser.add_argument("--slots", type=int, default=2431)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_usage_overlap(
            load_colibri_usage(args.training),
            load_colibri_usage(args.heldout),
            slots=args.slots,
        )
    except ColibriUsageAnalysisError as exc:
        print(f"colibri_usage_overlap: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
