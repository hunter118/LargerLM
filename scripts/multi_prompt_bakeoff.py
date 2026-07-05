#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from largerlm.result_summary import ResultSummaryError, result_bakeoff_files


SCHEMA = "largerlm.multi_prompt_bakeoff.v1"


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _task_by_prompt_variant(plan: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    tasks: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in _as_list(plan.get("tasks")):
        task = _as_mapping(raw)
        prompt = task.get("prompt_label")
        variant = task.get("variant")
        if isinstance(prompt, str) and isinstance(variant, str):
            tasks[(prompt, variant)] = task
    return tasks


def _prompt_labels(plan: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for raw in _as_list(plan.get("prompts")):
        label = _as_mapping(raw).get("label")
        if isinstance(label, str):
            labels.append(label)
    return labels


def _result_path(task: dict[str, Any] | None) -> str | None:
    if not task:
        return None
    path = task.get("result_path")
    return path if isinstance(path, str) and path else None


def _result_exists(task: dict[str, Any] | None) -> bool:
    path = _result_path(task)
    return path is not None and Path(path).exists()


def build_multi_prompt_bakeoff(
    plan: dict[str, Any],
    *,
    baseline_variant: str,
    candidate_variants: list[str],
    min_completed_prompts: int | None = None,
    top_limit: int = 12,
    system_slowdown_ratio: float = 1.8,
    promote_only_replay_files_ready: bool = True,
    allow_prefill_policy_change: bool = False,
) -> dict[str, Any]:
    if not candidate_variants:
        raise SystemExit("at least one candidate variant is required")
    prompts = _prompt_labels(plan)
    if min_completed_prompts is None:
        min_completed_prompts = len(prompts)
    if min_completed_prompts < 1:
        raise SystemExit("--min-completed-prompts must be positive")

    tasks = _task_by_prompt_variant(plan)
    prompt_results: list[dict[str, Any]] = []
    candidate_stats: dict[str, dict[str, int]] = {
        candidate: {
            "wins": 0,
            "ties": 0,
            "baseline_preferred": 0,
            "inconclusive": 0,
            "missing": 0,
        }
        for candidate in candidate_variants
    }

    for prompt in prompts:
        baseline_task = tasks.get((prompt, baseline_variant))
        candidate_tasks = {
            candidate: tasks.get((prompt, candidate))
            for candidate in candidate_variants
        }
        missing: list[dict[str, str]] = []
        if not _result_exists(baseline_task):
            missing.append(
                {
                    "variant": baseline_variant,
                    "path": _result_path(baseline_task) or "",
                }
            )
        for candidate, task in candidate_tasks.items():
            if not _result_exists(task):
                missing.append(
                    {
                        "variant": candidate,
                        "path": _result_path(task) or "",
                    }
                )
                candidate_stats[candidate]["missing"] += 1
        if missing:
            prompt_results.append(
                {
                    "prompt_label": prompt,
                    "status": "incomplete",
                    "missing_results": missing,
                }
            )
            continue

        baseline_path = _result_path(baseline_task)
        candidate_paths = [
            _result_path(candidate_tasks[candidate])
            for candidate in candidate_variants
        ]
        assert baseline_path is not None
        assert all(path is not None for path in candidate_paths)
        try:
            bakeoff = result_bakeoff_files(
                baseline_path,
                [str(path) for path in candidate_paths if path is not None],
                top_limit=top_limit,
                system_slowdown_ratio=system_slowdown_ratio,
                promote_only_replay_files_ready=promote_only_replay_files_ready,
                allow_prefill_policy_change=allow_prefill_policy_change,
            )
            status = "complete"
            error = None
        except ResultSummaryError as exc:
            bakeoff = None
            status = "error"
            error = str(exc)
        if bakeoff is None:
            prompt_results.append(
                {"prompt_label": prompt, "status": status, "error": error}
            )
            for candidate in candidate_variants:
                candidate_stats[candidate]["inconclusive"] += 1
            continue
        selected = _as_mapping(bakeoff.get("selected"))
        selected_path = selected.get("path")
        for candidate in _as_list(bakeoff.get("candidates")):
            row = _as_mapping(candidate)
            path = row.get("path")
            variant = next(
                (
                    name
                    for name in candidate_variants
                    if _result_path(candidate_tasks[name]) == path
                ),
                None,
            )
            if variant is None:
                continue
            decision = row.get("decision")
            if selected.get("role") == "candidate" and selected_path == path:
                candidate_stats[variant]["wins"] += 1
            elif decision == "tie":
                candidate_stats[variant]["ties"] += 1
            elif decision == "prefer_baseline":
                candidate_stats[variant]["baseline_preferred"] += 1
            else:
                candidate_stats[variant]["inconclusive"] += 1
        prompt_results.append(
            {
                "prompt_label": prompt,
                "status": "complete",
                "baseline_path": baseline_path,
                "candidate_paths": {
                    candidate: _result_path(candidate_tasks[candidate])
                    for candidate in candidate_variants
                },
                "selected": selected,
                "winner": bakeoff.get("winner"),
                "baseline_retained": bakeoff.get("baseline_retained"),
                "candidates": bakeoff.get("candidates"),
            }
        )

    completed = [row for row in prompt_results if row.get("status") == "complete"]
    complete_count = len(completed)
    if complete_count < min_completed_prompts:
        overall_decision = "incomplete"
        selected_variant = baseline_variant
        reasons = ["not_enough_completed_prompts"]
    else:
        selected_variant = baseline_variant
        overall_decision = "retain_baseline"
        reasons = ["no_candidate_won_all_completed_prompts"]
        for candidate, stats in candidate_stats.items():
            if stats["wins"] == complete_count:
                selected_variant = candidate
                overall_decision = "prefer_candidate"
                reasons = ["candidate_won_all_completed_prompts"]
                break

    return {
        "schema": SCHEMA,
        "plan_schema": plan.get("schema"),
        "baseline_variant": baseline_variant,
        "candidate_variants": candidate_variants,
        "prompt_count": len(prompts),
        "completed_prompt_count": complete_count,
        "min_completed_prompts": min_completed_prompts,
        "overall_decision": overall_decision,
        "selected_variant": selected_variant,
        "reasons": reasons,
        "candidate_stats": candidate_stats,
        "prompts": prompt_results,
        "promote_only_replay_files_ready": promote_only_replay_files_ready,
        "allow_prefill_policy_change": allow_prefill_policy_change,
    }


def format_multi_prompt_bakeoff_text(payload: dict[str, Any]) -> str:
    lines = [
        f"multi-prompt bakeoff: prompts={payload.get('prompt_count')} "
        f"complete={payload.get('completed_prompt_count')}/"
        f"{payload.get('min_completed_prompts')}",
        f"decision: {payload.get('overall_decision')} "
        f"selected={payload.get('selected_variant')} "
        f"reasons={','.join(str(item) for item in _as_list(payload.get('reasons')))}",
    ]
    for candidate, stats in _as_mapping(payload.get("candidate_stats")).items():
        if not isinstance(stats, dict):
            continue
        lines.append(
            f"candidate {candidate}: wins={stats.get('wins', 0)} "
            f"ties={stats.get('ties', 0)} "
            f"baseline_preferred={stats.get('baseline_preferred', 0)} "
            f"inconclusive={stats.get('inconclusive', 0)} "
            f"missing={stats.get('missing', 0)}"
        )
    for prompt in _as_list(payload.get("prompts")):
        row = _as_mapping(prompt)
        label = row.get("prompt_label")
        status = row.get("status")
        if status == "complete":
            selected = _as_mapping(row.get("selected"))
            lines.append(
                f"  complete: {label} selected={selected.get('role')} "
                f"path={selected.get('path')}"
            )
        elif status == "incomplete":
            missing = [
                _as_mapping(item).get("variant")
                for item in _as_list(row.get("missing_results"))
            ]
            lines.append(f"  incomplete: {label} missing={','.join(str(item) for item in missing)}")
        else:
            lines.append(f"  {status}: {label} error={row.get('error')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-prompt result-bakeoff decisions from a multi-prompt replay plan."
    )
    parser.add_argument("plan", help="largerlm.multi_prompt_replay_plan.v1 JSON")
    parser.add_argument("--baseline-variant", required=True)
    parser.add_argument("--candidate-variant", action="append", required=True)
    parser.add_argument("--min-completed-prompts", type=int, default=None)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--system-slowdown-ratio", type=float, default=1.8)
    parser.add_argument(
        "--allow-prefill-policy-change",
        action="store_true",
    )
    parser.add_argument(
        "--no-promote-only-replay-files-ready",
        action="store_true",
    )
    parser.add_argument("--write-result", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = build_multi_prompt_bakeoff(
        _load_json(Path(args.plan)),
        baseline_variant=args.baseline_variant,
        candidate_variants=args.candidate_variant,
        min_completed_prompts=args.min_completed_prompts,
        top_limit=args.top,
        system_slowdown_ratio=args.system_slowdown_ratio,
        promote_only_replay_files_ready=not args.no_promote_only_replay_files_ready,
        allow_prefill_policy_change=args.allow_prefill_policy_change,
    )
    if args.write_result:
        path = Path(args.write_result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_multi_prompt_bakeoff_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
