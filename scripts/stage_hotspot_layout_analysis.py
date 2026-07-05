#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _nonnegative_int(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _token_result(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("token_result"), dict):
        return payload["token_result"]
    response = _as_mapping(payload.get("response"))
    largerlm = _as_mapping(response.get("largerlm"))
    token_result = _as_mapping(largerlm.get("token_result"))
    if token_result:
        return token_result
    return payload


def _expert_counts_by_layer(path: Path | None) -> dict[int, int]:
    if path is None:
        return {}
    payload = _load_json(path)
    counts: dict[int, int] = {}
    for item in _as_list(payload.get("layers")):
        layer = _as_mapping(item)
        layer_id = _nonnegative_int(layer.get("layer"))
        count = _nonnegative_int(layer.get("num_experts"))
        if layer_id is not None and count is not None:
            counts[layer_id] = count
    return counts


def _selected_experts(row: dict[str, Any]) -> list[int]:
    experts: list[int] = []
    for item in _as_list(row.get("selected_experts")):
        parsed = _nonnegative_int(item)
        if parsed is None:
            return []
        experts.append(parsed)
    return experts


def _current_order_runs(experts: list[int]) -> list[list[int]]:
    if not experts:
        return []
    runs: list[list[int]] = [[experts[0]]]
    for expert in experts[1:]:
        if expert == runs[-1][-1] + 1:
            runs[-1].append(expert)
        else:
            runs.append([expert])
    return runs


def _range_count_for_order(experts: list[int], position_by_expert: dict[int, int]) -> int:
    positions = sorted(
        position_by_expert[expert]
        for expert in experts
        if expert in position_by_expert
    )
    if not positions:
        return 0
    runs = 1
    for left, right in zip(positions, positions[1:]):
        if right != left + 1:
            runs += 1
    return runs


def _analyze_hotspot(row: dict[str, Any], *, source: str, rank: int) -> dict[str, Any]:
    experts = _selected_experts(row)
    gaps = [right - left for left, right in zip(experts, experts[1:])]
    runs = _current_order_runs(experts)
    coalesced_range_count = _nonnegative_int(row.get("coalesced_range_count")) or 0
    ideal_clustered_range_count = 1 if experts else 0
    range_reduction = max(0, coalesced_range_count - ideal_clustered_range_count)
    return {
        "source": source,
        "rank": rank,
        "chunk_index": _nonnegative_int(row.get("chunk_index")),
        "layer": _nonnegative_int(row.get("layer")),
        "tile_index": _nonnegative_int(row.get("tile_index")),
        "copy_elapsed_seconds": _finite_float(row.get("copy_elapsed_seconds")),
        "planned_read_bytes": _nonnegative_int(row.get("planned_read_bytes")),
        "copy_read_calls": _nonnegative_int(row.get("copy_read_calls")),
        "raw_range_count": _nonnegative_int(row.get("raw_range_count")),
        "coalesced_range_count": coalesced_range_count,
        "selected_expert_count": len(experts),
        "selected_experts": experts,
        "current_order_run_count": len(runs),
        "current_order_run_lengths": [len(run) for run in runs],
        "current_order_contiguous_pair_count": sum(1 for gap in gaps if gap == 1),
        "current_order_max_gap": max(gaps) if gaps else None,
        "current_order_mean_gap": (sum(gaps) / len(gaps)) if gaps else None,
        "ideal_clustered_range_count": ideal_clustered_range_count,
        "ideal_clustered_range_reduction": range_reduction,
        "ideal_clustered_range_reduction_fraction": (
            range_reduction / coalesced_range_count if coalesced_range_count else None
        ),
    }


def _hotspot_rows_for_result(path: Path) -> tuple[int | None, list[dict[str, Any]]]:
    payload = _load_json(path)
    token_result = _token_result(payload)
    actual = _as_mapping(token_result.get("prefill_actual_read_time"))
    rows_by_stage: dict[tuple[int | None, int | None, int | None], dict[str, Any]] = {}
    for source, field in (
        ("copy", "expert_stage_copy_hotspots"),
        ("ranges", "expert_stage_range_hotspots"),
    ):
        for rank, raw_row in enumerate(_as_list(actual.get(field)), start=1):
            row = _as_mapping(raw_row)
            if not row:
                continue
            analyzed = _analyze_hotspot(row, source=source, rank=rank)
            key = (
                analyzed.get("chunk_index"),
                analyzed.get("layer"),
                analyzed.get("tile_index"),
            )
            source_rank = {"source": source, "rank": rank}
            existing = rows_by_stage.get(key)
            if existing is None:
                analyzed["sources"] = [source_rank]
                rows_by_stage[key] = analyzed
            else:
                existing.setdefault("sources", []).append(source_rank)
                if (
                    (analyzed.get("copy_elapsed_seconds") or 0.0)
                    > (existing.get("copy_elapsed_seconds") or 0.0)
                ):
                    existing["copy_elapsed_seconds"] = analyzed.get(
                        "copy_elapsed_seconds"
                    )
                if (
                    (analyzed.get("coalesced_range_count") or 0)
                    > (existing.get("coalesced_range_count") or 0)
                ):
                    existing["coalesced_range_count"] = analyzed.get(
                        "coalesced_range_count"
                    )
    rows = list(rows_by_stage.values())
    rows.sort(
        key=lambda item: (
            item.get("ideal_clustered_range_reduction") or 0,
            item.get("copy_elapsed_seconds") or 0.0,
            item.get("planned_read_bytes") or 0,
        ),
        reverse=True,
    )
    return actual.get("expert_stage_io_stage_count"), rows


def _candidate_observed_order(
    observed_experts: set[int],
    pair_weights: dict[tuple[int, int], float],
    expert_weights: dict[int, float],
) -> list[int]:
    remaining = set(observed_experts)
    if not remaining:
        return []
    first = min(
        remaining,
        key=lambda expert: (-expert_weights.get(expert, 0.0), expert),
    )
    order = [first]
    remaining.remove(first)
    while remaining:
        left = order[0]
        right = order[-1]
        best: tuple[float, int, int, str] | None = None
        for expert in remaining:
            left_key = tuple(sorted((expert, left)))
            right_key = tuple(sorted((expert, right)))
            left_score = pair_weights.get(left_key, 0.0)
            right_score = pair_weights.get(right_key, 0.0)
            for side, score in (("left", left_score), ("right", right_score)):
                candidate = (
                    score,
                    int(expert_weights.get(expert, 0.0) * 1_000_000),
                    -expert,
                    side,
                )
                if best is None or candidate > best:
                    best = candidate
        assert best is not None
        expert = -best[2]
        if best[3] == "left":
            order.insert(0, expert)
        else:
            order.append(expert)
        remaining.remove(expert)
    return order


def _layer_coactivation_candidate(
    *,
    layer: int,
    rows: list[dict[str, Any]],
    num_experts: int,
) -> dict[str, Any]:
    pair_weights: dict[tuple[int, int], float] = {}
    expert_weights: dict[int, float] = {}
    observed: set[int] = set()
    for row in rows:
        experts = _selected_experts(row)
        if not experts:
            continue
        weight = _finite_float(row.get("copy_elapsed_seconds")) or 1.0
        for expert in experts:
            observed.add(expert)
            expert_weights[expert] = expert_weights.get(expert, 0.0) + weight
        for left, right in combinations(sorted(set(experts)), 2):
            key = (left, right)
            pair_weights[key] = pair_weights.get(key, 0.0) + weight
    observed_order = _candidate_observed_order(observed, pair_weights, expert_weights)
    unobserved = [expert for expert in range(num_experts) if expert not in observed]
    expert_order = observed_order + unobserved
    position_by_expert = {expert: index for index, expert in enumerate(expert_order)}
    simulated_rows = []
    total_current = 0
    total_candidate = 0
    total_copy_seconds = 0.0
    for row in rows:
        experts = _selected_experts(row)
        current = _nonnegative_int(row.get("coalesced_range_count"))
        if current is None:
            current = _range_count_for_order(experts, {expert: expert for expert in experts})
        candidate = _range_count_for_order(experts, position_by_expert)
        total_current += current
        total_candidate += candidate
        total_copy_seconds += _finite_float(row.get("copy_elapsed_seconds")) or 0.0
        simulated_rows.append(
            {
                "chunk_index": row.get("chunk_index"),
                "tile_index": row.get("tile_index"),
                "sources": row.get("sources", []),
                "selected_expert_count": len(experts),
                "current_range_count": current,
                "candidate_range_count": candidate,
                "range_reduction": max(0, current - candidate),
                "copy_elapsed_seconds": row.get("copy_elapsed_seconds"),
            }
        )
    return {
        "layer": layer,
        "num_experts": num_experts,
        "sample_count": len(rows),
        "observed_expert_count": len(observed_order),
        "observed_expert_order": observed_order,
        "unobserved_expert_count": len(unobserved),
        "expert_order": expert_order,
        "total_current_range_count": total_current,
        "total_candidate_range_count": total_candidate,
        "total_range_reduction": max(0, total_current - total_candidate),
        "total_range_reduction_fraction": (
            (total_current - total_candidate) / total_current
            if total_current > 0
            else None
        ),
        "total_copy_elapsed_seconds": total_copy_seconds,
        "simulated_rows": simulated_rows,
    }


def _coactivation_candidates(
    rows: list[dict[str, Any]],
    *,
    expert_counts_by_layer: dict[int, int],
    top_layer_limit: int,
) -> list[dict[str, Any]]:
    rows_by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        layer = _nonnegative_int(row.get("layer"))
        if layer is None:
            continue
        rows_by_layer.setdefault(layer, []).append(row)
    candidates = []
    for layer, layer_rows in rows_by_layer.items():
        observed_experts = {
            expert
            for row in layer_rows
            for expert in _selected_experts(row)
        }
        if not observed_experts:
            continue
        num_experts = expert_counts_by_layer.get(layer, max(observed_experts) + 1)
        candidates.append(
            _layer_coactivation_candidate(
                layer=layer,
                rows=layer_rows,
                num_experts=num_experts,
            )
        )
    candidates.sort(
        key=lambda item: (
            item.get("total_range_reduction") or 0,
            item.get("total_copy_elapsed_seconds") or 0.0,
            item.get("observed_expert_count") or 0,
        ),
        reverse=True,
    )
    return candidates[:top_layer_limit]


def analyze_results(
    paths: list[Path],
    *,
    expert_layout_path: Path | None = None,
    top_layer_limit: int = 10,
) -> dict[str, Any]:
    per_result = []
    all_rows: list[dict[str, Any]] = []
    for path in paths:
        stage_count, rows = _hotspot_rows_for_result(path)
        for row in rows:
            row["source_result"] = str(path)
        per_result.append(
            {
                "source_result": str(path),
                "source_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "stage_count": stage_count,
                "row_count": len(rows),
                "top_layout_targets": rows[:5],
            }
        )
        all_rows.extend(rows)
    all_rows.sort(
        key=lambda item: (
            item.get("ideal_clustered_range_reduction") or 0,
            item.get("copy_elapsed_seconds") or 0.0,
            item.get("planned_read_bytes") or 0,
        ),
        reverse=True,
    )
    expert_counts = _expert_counts_by_layer(expert_layout_path)
    return {
        "schema": "largerlm.stage_hotspot_layout_analysis.v2",
        "source_results": per_result,
        "source_result_count": len(paths),
        "expert_layout": str(expert_layout_path) if expert_layout_path is not None else None,
        "row_count": len(all_rows),
        "rows": all_rows,
        "top_layout_targets": all_rows[:5],
        "coactivation_candidates": _coactivation_candidates(
            all_rows,
            expert_counts_by_layer=expert_counts,
            top_layer_limit=top_layer_limit,
        ),
    }


def analyze_result(path: Path) -> dict[str, Any]:
    return analyze_results([path])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze selected-expert locality in routed stage hotspots."
    )
    parser.add_argument("result_json", type=Path, nargs="+")
    parser.add_argument("--expert-layout", type=Path)
    parser.add_argument("--top-layer-limit", type=int, default=10)
    parser.add_argument("--write-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.top_layer_limit <= 0:
        raise SystemExit("--top-layer-limit must be positive")
    analysis = analyze_results(
        args.result_json,
        expert_layout_path=args.expert_layout,
        top_layer_limit=args.top_layer_limit,
    )
    text = json.dumps(analysis, indent=2, sort_keys=True)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(text + "\n", encoding="utf-8")
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
