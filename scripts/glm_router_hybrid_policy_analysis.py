#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OBSERVED_LOGIT_DRIFT = 5.4836273193359375e-06
DEFAULT_CUSTOM_TO_MPSGRAPH_RATIO = 0.043683 / 0.05737041700000001
DEFAULT_SAFETY_MULTIPLIER = 4.0


@dataclass(frozen=True)
class RouterLayerRecord:
    chunk_index: int
    layer: int
    token_count: int
    min_effective_score_margin: float
    min_topk_score_margin: float | None
    mean_effective_score_margin: float | None
    router_elapsed_seconds: float
    effective_near_tie_counts: dict[str, int]
    topk_near_tie_counts: dict[str, int]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _finite_float(value: object, *, field: str) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise SystemExit(f"{field} must be finite")
    return result


def _positive_float(value: object, *, field: str) -> float:
    result = _finite_float(value, field=field)
    if result is None or result <= 0.0:
        raise SystemExit(f"{field} must be a positive finite number")
    return result


def _positive_int(value: object, *, field: str) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        raise SystemExit(f"{field} must be positive")
    return int(value)


def _near_tie_counts(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise SystemExit(f"{field}.{key} must be a non-negative integer")
        counts[key] = int(raw)
    return counts


def _threshold_key(threshold: float) -> str:
    return f"le_{threshold:.0e}"


def _threshold_from_key(key: str) -> float | None:
    if not key.startswith("le_"):
        return None
    try:
        threshold = float(key[3:])
    except ValueError:
        return None
    return threshold if math.isfinite(threshold) and threshold > 0.0 else None


def _matching_summary_key(
    records: list[RouterLayerRecord],
    threshold: float,
    *,
    count_attr: str,
) -> str | None:
    keys: set[str] = set()
    for record in records:
        counts = getattr(record, count_attr)
        if isinstance(counts, dict):
            keys.update(counts)
    for key in sorted(keys):
        key_threshold = _threshold_from_key(key)
        if key_threshold is not None and math.isclose(
            key_threshold,
            threshold,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            return key
    return None


def _parse_float_list(value: str, *, field: str) -> tuple[float, ...]:
    items: list[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            item = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{field} values must be floats") from exc
        if not math.isfinite(item) or item <= 0.0:
            raise argparse.ArgumentTypeError(f"{field} values must be positive finite floats")
        items.append(item)
    if not items:
        raise argparse.ArgumentTypeError(f"{field} must not be empty")
    return tuple(items)


def _parse_multipliers(value: str) -> tuple[float, ...]:
    return _parse_float_list(value, field="safety multipliers")


def _parse_thresholds(value: str) -> tuple[float, ...]:
    return _parse_float_list(value, field="absolute thresholds")


def _iter_router_records(payload: dict[str, Any]) -> list[RouterLayerRecord]:
    prompt_prefill = (
        payload.get("token_result", {})
        if isinstance(payload.get("token_result"), dict)
        else {}
    ).get("prompt_prefill")
    if not isinstance(prompt_prefill, dict):
        raise SystemExit("result JSON does not contain token_result.prompt_prefill")
    chunks = prompt_prefill.get("chunks")
    if not isinstance(chunks, list):
        raise SystemExit("result JSON does not contain prompt_prefill.chunks")

    records: list[RouterLayerRecord] = []
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        layers = chunk.get("layers")
        if not isinstance(layers, list):
            continue
        for layer_index, layer_entry in enumerate(layers):
            if not isinstance(layer_entry, dict):
                continue
            staged_mlp = layer_entry.get("staged_mlp")
            if not isinstance(staged_mlp, dict):
                continue
            margin = staged_mlp.get("router_margin_summary")
            if not isinstance(margin, dict):
                continue
            min_effective = _finite_float(
                margin.get("min_effective_score_margin"),
                field=f"chunk {chunk_index} layer {layer_index} min_effective_score_margin",
            )
            token_count = _positive_int(
                margin.get("token_count"),
                field=f"chunk {chunk_index} layer {layer_index} token_count",
            )
            if min_effective is None or token_count is None:
                continue
            router_gate = staged_mlp.get("router_gate_proj")
            router_elapsed = 0.0
            if isinstance(router_gate, dict):
                elapsed = _finite_float(
                    router_gate.get("elapsed_seconds"),
                    field=f"chunk {chunk_index} layer {layer_index} router elapsed_seconds",
                )
                if elapsed is not None:
                    router_elapsed = max(0.0, elapsed)
            layer_id = layer_entry.get("layer")
            if isinstance(layer_id, bool) or not isinstance(layer_id, int):
                layer_id = layer_index
            records.append(
                RouterLayerRecord(
                    chunk_index=chunk_index,
                    layer=int(layer_id),
                    token_count=token_count,
                    min_effective_score_margin=min_effective,
                    min_topk_score_margin=_finite_float(
                        margin.get("min_topk_score_margin"),
                        field=f"chunk {chunk_index} layer {layer_index} min_topk_score_margin",
                    ),
                    mean_effective_score_margin=_finite_float(
                        margin.get("mean_effective_score_margin"),
                        field=(
                            f"chunk {chunk_index} layer {layer_index} "
                            "mean_effective_score_margin"
                        ),
                    ),
                    router_elapsed_seconds=router_elapsed,
                    effective_near_tie_counts=_near_tie_counts(
                        margin.get("effective_near_tie_counts"),
                        field=f"chunk {chunk_index} layer {layer_index} effective_near_tie_counts",
                    ),
                    topk_near_tie_counts=_near_tie_counts(
                        margin.get("topk_near_tie_counts"),
                        field=f"chunk {chunk_index} layer {layer_index} topk_near_tie_counts",
                    ),
                )
            )
    if not records:
        raise SystemExit("no routed layer router_margin_summary records found")
    return records


def _json_object_from_line(first_line: str, handle: Any) -> dict[str, Any]:
    start = first_line.find("{")
    if start < 0:
        raise SystemExit("streamed JSON object line is missing an opening brace")
    parts: list[str] = []
    balance = 0
    line = first_line[start:]
    while True:
        balance += line.count("{") - line.count("}")
        if balance <= 0:
            end = line.rfind("}")
            if end < 0:
                raise SystemExit("streamed JSON object ended without a closing brace")
            parts.append(line[: end + 1])
            break
        parts.append(line)
        next_line = handle.readline()
        if next_line == "":
            raise SystemExit("streamed JSON object reached EOF before closing brace")
        line = next_line
    try:
        payload = json.loads("".join(parts))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse streamed JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("streamed JSON object must parse to an object")
    return payload


def _iter_router_records_stream(path: Path) -> list[RouterLayerRecord]:
    records: list[RouterLayerRecord] = []
    current_chunk: int | None = None
    current_layer: int | None = None
    current_router_elapsed = 0.0

    try:
        handle_cm = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc

    with handle_cm as handle:
        while True:
            line = handle.readline()
            if line == "":
                break
            stripped = line.strip()
            if stripped.startswith('"chunk_index":'):
                raw = stripped.split(":", 1)[1].rstrip(",")
                try:
                    current_chunk = int(raw)
                except ValueError:
                    pass
                continue
            if stripped.startswith('"layer":'):
                raw = stripped.split(":", 1)[1].rstrip(",")
                try:
                    current_layer = int(raw)
                except ValueError:
                    pass
                continue
            if stripped.startswith('"router_gate_proj":'):
                router_gate = _json_object_from_line(line, handle)
                elapsed = _finite_float(
                    router_gate.get("elapsed_seconds"),
                    field=f"streamed router_gate_proj[{len(records)}].elapsed_seconds",
                )
                current_router_elapsed = max(0.0, elapsed or 0.0)
                layer_value = router_gate.get("layer")
                if not isinstance(layer_value, bool) and isinstance(layer_value, int):
                    current_layer = int(layer_value)
                continue
            if not stripped.startswith('"router_margin_summary":'):
                continue

            margin = _json_object_from_line(line, handle)
            min_effective = _finite_float(
                margin.get("min_effective_score_margin"),
                field=f"streamed router_margin_summary[{len(records)}].min_effective_score_margin",
            )
            token_count = _positive_int(
                margin.get("token_count"),
                field=f"streamed router_margin_summary[{len(records)}].token_count",
            )
            if min_effective is None or token_count is None:
                continue
            layer = current_layer if current_layer is not None else len(records)
            chunk = current_chunk if current_chunk is not None else 0
            records.append(
                RouterLayerRecord(
                    chunk_index=int(chunk),
                    layer=int(layer),
                    token_count=token_count,
                    min_effective_score_margin=min_effective,
                    min_topk_score_margin=_finite_float(
                        margin.get("min_topk_score_margin"),
                        field=(
                            f"streamed router_margin_summary[{len(records)}]."
                            "min_topk_score_margin"
                        ),
                    ),
                    mean_effective_score_margin=_finite_float(
                        margin.get("mean_effective_score_margin"),
                        field=(
                            f"streamed router_margin_summary[{len(records)}]."
                            "mean_effective_score_margin"
                        ),
                    ),
                    router_elapsed_seconds=current_router_elapsed,
                    effective_near_tie_counts=_near_tie_counts(
                        margin.get("effective_near_tie_counts"),
                        field=(
                            f"streamed router_margin_summary[{len(records)}]."
                            "effective_near_tie_counts"
                        ),
                    ),
                    topk_near_tie_counts=_near_tie_counts(
                        margin.get("topk_near_tie_counts"),
                        field=(
                            f"streamed router_margin_summary[{len(records)}]."
                            "topk_near_tie_counts"
                        ),
                    ),
                )
            )
            current_router_elapsed = 0.0
    if not records:
        raise SystemExit("no streamed routed layer router_margin_summary records found")
    return records


def _read_consistency_observations(path: Path) -> dict[str, float]:
    payload = _load_json(path)
    comparisons = payload.get("comparisons")
    drift: float | None = None
    if isinstance(comparisons, dict):
        custom_vs_mps = comparisons.get("custom_resident_vs_mpsgraph_resident")
        if isinstance(custom_vs_mps, dict):
            drift = _finite_float(
                custom_vs_mps.get("logits_max_abs"),
                field="custom_resident_vs_mpsgraph_resident.logits_max_abs",
            )
    custom = payload.get("custom_resident_linear")
    mpsgraph = payload.get("mpsgraph_resident_linear")
    ratio: float | None = None
    if isinstance(custom, dict) and isinstance(mpsgraph, dict):
        custom_elapsed = _finite_float(
            custom.get("elapsed_seconds"),
            field="custom_resident_linear.elapsed_seconds",
        )
        mpsgraph_elapsed = _finite_float(
            mpsgraph.get("elapsed_seconds"),
            field="mpsgraph_resident_linear.elapsed_seconds",
        )
        if (
            custom_elapsed is not None
            and custom_elapsed > 0.0
            and mpsgraph_elapsed is not None
            and mpsgraph_elapsed > 0.0
        ):
            ratio = custom_elapsed / mpsgraph_elapsed
    result: dict[str, float] = {}
    if drift is not None and drift > 0.0:
        result["observed_logit_drift"] = drift
    if ratio is not None and ratio > 0.0:
        result["custom_to_mpsgraph_router_ratio"] = ratio
    return result


def _layer_ref(record: RouterLayerRecord) -> dict[str, int]:
    return {"chunk_index": record.chunk_index, "layer": record.layer}


def analyze_router_hybrid_policy_records(
    records: list[RouterLayerRecord],
    *,
    observed_logit_drift: float,
    safety_multipliers: tuple[float, ...],
    absolute_thresholds: tuple[float, ...],
    promotion_safety_multiplier: float,
    custom_to_mpsgraph_router_ratio: float,
) -> dict[str, Any]:
    observed_logit_drift = _positive_float(
        observed_logit_drift,
        field="observed_logit_drift",
    )
    promotion_safety_multiplier = _positive_float(
        promotion_safety_multiplier,
        field="promotion_safety_multiplier",
    )
    custom_to_mpsgraph_router_ratio = _positive_float(
        custom_to_mpsgraph_router_ratio,
        field="custom_to_mpsgraph_router_ratio",
    )
    if not records:
        raise SystemExit("no router records to analyze")
    min_effective = min(record.min_effective_score_margin for record in records)
    min_topk_values = [
        record.min_topk_score_margin
        for record in records
        if record.min_topk_score_margin is not None
    ]
    weakest = min(records, key=lambda record: record.min_effective_score_margin)
    router_elapsed_total = sum(record.router_elapsed_seconds for record in records)
    token_layer_count = sum(record.token_count for record in records)
    threshold_values: dict[float, set[str]] = {}
    for multiplier in safety_multipliers:
        threshold_values.setdefault(observed_logit_drift * multiplier, set()).add(
            f"drift_x_{multiplier:g}"
        )
    for threshold in absolute_thresholds:
        threshold_values.setdefault(threshold, set()).add("absolute")

    thresholds: list[dict[str, Any]] = []
    for threshold, sources in sorted(threshold_values.items()):
        fallback = [
            record
            for record in records
            if record.min_effective_score_margin <= threshold
        ]
        safe = [
            record
            for record in records
            if record.min_effective_score_margin > threshold
        ]
        fallback_elapsed = sum(record.router_elapsed_seconds for record in fallback)
        safe_elapsed = sum(record.router_elapsed_seconds for record in safe)
        custom_all_elapsed_estimate = (
            router_elapsed_total * custom_to_mpsgraph_router_ratio
        )
        static_policy_elapsed = (
            fallback_elapsed
            + safe_elapsed * custom_to_mpsgraph_router_ratio
        )
        online_layer_fallback_elapsed = custom_all_elapsed_estimate + fallback_elapsed
        key = _threshold_key(threshold)
        effective_summary_key = _matching_summary_key(
            records,
            threshold,
            count_attr="effective_near_tie_counts",
        )
        topk_summary_key = _matching_summary_key(
            records,
            threshold,
            count_attr="topk_near_tie_counts",
        )
        effective_near_tie = (
            sum(
                record.effective_near_tie_counts.get(effective_summary_key, 0)
                for record in records
            )
            if effective_summary_key is not None
            else None
        )
        topk_near_tie = (
            sum(
                record.topk_near_tie_counts.get(topk_summary_key, 0)
                for record in records
            )
            if topk_summary_key is not None
            else None
        )
        thresholds.append(
            {
                "threshold": threshold,
                "sources": sorted(sources),
                "threshold_key": key,
                "effective_near_tie_summary_key": effective_summary_key,
                "topk_near_tie_summary_key": topk_summary_key,
                "safe_layer_count": len(safe),
                "fallback_layer_count": len(fallback),
                "fallback_layer_fraction": len(fallback) / len(records),
                "fallback_layers": [_layer_ref(record) for record in fallback[:24]],
                "fallback_layers_truncated": len(fallback) > 24,
                "effective_near_tie_token_layer_count": effective_near_tie,
                "topk_near_tie_token_layer_count": topk_near_tie,
                "fallback_token_layer_count_lower_bound": len(fallback),
                "router_elapsed_seconds": {
                    "current_mpsgraph_total": router_elapsed_total,
                    "fallback_mpsgraph_subset": fallback_elapsed,
                    "safe_mpsgraph_subset": safe_elapsed,
                    "custom_all_estimate": custom_all_elapsed_estimate,
                    "static_layer_policy_estimate": static_policy_elapsed,
                    "static_layer_policy_savings": router_elapsed_total
                    - static_policy_elapsed,
                    "online_custom_first_layer_fallback_estimate": (
                        online_layer_fallback_elapsed
                    ),
                    "online_custom_first_layer_fallback_savings": (
                        router_elapsed_total - online_layer_fallback_elapsed
                    ),
                },
            }
        )

    promotion_threshold = observed_logit_drift * promotion_safety_multiplier
    promotion_row = min(
        thresholds,
        key=lambda row: abs(float(row["threshold"]) - promotion_threshold),
    )
    global_custom_promotable = min_effective > promotion_threshold
    safe_layer_count_at_promotion = sum(
        1 for record in records if record.min_effective_score_margin > promotion_threshold
    )
    fallback_layer_count_at_promotion = len(records) - safe_layer_count_at_promotion
    return {
        "schema": "largerlm.glm_router_hybrid_policy_analysis.v1",
        "inputs": {
            "observed_logit_drift": observed_logit_drift,
            "promotion_safety_multiplier": promotion_safety_multiplier,
            "promotion_threshold": promotion_threshold,
            "custom_to_mpsgraph_router_ratio": custom_to_mpsgraph_router_ratio,
        },
        "summary": {
            "routed_layer_count": len(records),
            "token_layer_count": token_layer_count,
            "router_elapsed_seconds_total": router_elapsed_total,
            "min_effective_score_margin": min_effective,
            "min_topk_score_margin": min(min_topk_values) if min_topk_values else None,
            "weakest_layer": {
                **_layer_ref(weakest),
                "min_effective_score_margin": weakest.min_effective_score_margin,
                "min_topk_score_margin": weakest.min_topk_score_margin,
                "token_count": weakest.token_count,
            },
        },
        "thresholds": thresholds,
        "recommendation": {
            "default_router_gate_backend": "mpsgraph-f32",
            "global_custom_promotable": global_custom_promotable,
            "global_custom_blocked_reason": (
                None
                if global_custom_promotable
                else (
                    "minimum effective router margin is not above observed "
                    "custom-vs-MPSGraph drift times the safety multiplier"
                )
            ),
            "layer_hybrid_candidate": (
                safe_layer_count_at_promotion > 0
                and fallback_layer_count_at_promotion > 0
            ),
            "safe_layer_count_at_promotion_threshold": safe_layer_count_at_promotion,
            "fallback_layer_count_at_promotion_threshold": fallback_layer_count_at_promotion,
            "online_layer_fallback_expected_to_help": (
                float(
                    promotion_row["router_elapsed_seconds"][
                        "online_custom_first_layer_fallback_savings"
                    ]
                )
                > 0.0
            ),
            "token_level_hybrid_needs_retained_router_logits": True,
            "notes": [
                "Static layer-policy savings are an upper bound from replay telemetry.",
                "Online custom-first layer fallback includes the cost of computing custom logits before rerunning MPSGraph for unsafe layers.",
                "Token-level fallback cannot be validated from this compact result because per-token router JSON/logits were not retained.",
            ],
        },
    }


def analyze_router_hybrid_policy(
    payload: dict[str, Any],
    *,
    observed_logit_drift: float,
    safety_multipliers: tuple[float, ...],
    absolute_thresholds: tuple[float, ...],
    promotion_safety_multiplier: float,
    custom_to_mpsgraph_router_ratio: float,
) -> dict[str, Any]:
    return analyze_router_hybrid_policy_records(
        _iter_router_records(payload),
        observed_logit_drift=observed_logit_drift,
        safety_multipliers=safety_multipliers,
        absolute_thresholds=absolute_thresholds,
        promotion_safety_multiplier=promotion_safety_multiplier,
        custom_to_mpsgraph_router_ratio=custom_to_mpsgraph_router_ratio,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze GLM router margin telemetry for a safe custom-Metal/MPSGraph "
            "hybrid router-gate policy without launching model kernels."
        )
    )
    parser.add_argument("result_json", type=Path)
    parser.add_argument(
        "--stream-result",
        action="store_true",
        help="Extract router telemetry from result_json without loading the full file.",
    )
    parser.add_argument("--consistency-json", type=Path, default=None)
    parser.add_argument(
        "--observed-logit-drift",
        type=float,
        default=DEFAULT_OBSERVED_LOGIT_DRIFT,
        help="Observed max logit drift between custom Metal and MPSGraph router gate.",
    )
    parser.add_argument(
        "--custom-to-mpsgraph-router-ratio",
        type=float,
        default=DEFAULT_CUSTOM_TO_MPSGRAPH_RATIO,
        help="Estimated custom router elapsed divided by MPSGraph router elapsed.",
    )
    parser.add_argument(
        "--safety-multipliers",
        type=_parse_multipliers,
        default=_parse_multipliers("1,2,4,8,16,32,64,128"),
    )
    parser.add_argument(
        "--absolute-thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("1e-6,1e-5,1e-4,1e-3"),
    )
    parser.add_argument(
        "--promotion-safety-multiplier",
        type=float,
        default=DEFAULT_SAFETY_MULTIPLIER,
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    observed_logit_drift = args.observed_logit_drift
    custom_to_mpsgraph_router_ratio = args.custom_to_mpsgraph_router_ratio
    drift_source = "cli-default"
    ratio_source = "cli-default"
    if args.consistency_json is not None:
        observations = _read_consistency_observations(args.consistency_json)
        if "observed_logit_drift" in observations:
            observed_logit_drift = observations["observed_logit_drift"]
            drift_source = str(args.consistency_json)
        if "custom_to_mpsgraph_router_ratio" in observations:
            custom_to_mpsgraph_router_ratio = observations[
                "custom_to_mpsgraph_router_ratio"
            ]
            ratio_source = str(args.consistency_json)

    if args.stream_result:
        analysis = analyze_router_hybrid_policy_records(
            _iter_router_records_stream(args.result_json),
            observed_logit_drift=observed_logit_drift,
            safety_multipliers=args.safety_multipliers,
            absolute_thresholds=args.absolute_thresholds,
            promotion_safety_multiplier=args.promotion_safety_multiplier,
            custom_to_mpsgraph_router_ratio=custom_to_mpsgraph_router_ratio,
        )
    else:
        analysis = analyze_router_hybrid_policy(
            _load_json(args.result_json),
            observed_logit_drift=observed_logit_drift,
            safety_multipliers=args.safety_multipliers,
            absolute_thresholds=args.absolute_thresholds,
            promotion_safety_multiplier=args.promotion_safety_multiplier,
            custom_to_mpsgraph_router_ratio=custom_to_mpsgraph_router_ratio,
        )
    analysis["inputs"]["result_json"] = str(args.result_json)
    analysis["inputs"]["observed_logit_drift_source"] = drift_source
    analysis["inputs"]["custom_to_mpsgraph_router_ratio_source"] = ratio_source
    text = json.dumps(analysis, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
