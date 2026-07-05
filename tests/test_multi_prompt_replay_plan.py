from __future__ import annotations

import json
from pathlib import Path

from scripts.multi_prompt_replay_plan import Variant, build_plan, extract_prompt_record
from scripts.multi_prompt_bakeoff import build_multi_prompt_bakeoff


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audit(path: Path) -> Path:
    return _write_json(
        path,
        {
            "launch_audit": {"ok": True},
            "request_check": {"ok": True},
            "applied_launch_profile": {
                "argv_safe_to_replay": True,
                "locked": True,
                "matches_prepared": True,
            },
        },
    )


def test_extract_prompt_record_from_prepared_result(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "prepared.json",
        {
            "request": {"prompt_token_ids": [1, 2, 3], "max_new_tokens": 1},
            "token_result": {"generated_token_ids": [4]},
        },
    )

    record = extract_prompt_record(path)

    assert record["prompt_token_ids"] == [1, 2, 3]
    assert record["prompt_token_count"] == 3
    assert record["source_generated_token_ids"] == [4]
    assert record["source_max_new_tokens"] == 1


def test_extract_prompt_record_from_http_raw_response(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "http.json",
        {
            "request": {"max_tokens": 2},
            "raw_response": json.dumps(
                {
                    "largerlm": {
                        "token_result": {
                            "prompt_token_ids": [10, 11],
                            "generated_token_ids": [12, 13],
                        }
                    }
                }
            ),
        },
    )

    record = extract_prompt_record(path)

    assert record["prompt_token_ids"] == [10, 11]
    assert record["source_generated_token_ids"] == [12, 13]
    assert record["source_max_new_tokens"] == 2


def test_build_plan_marks_locked_variants_ready_and_dedupes_prompts(tmp_path: Path) -> None:
    prompt_a = _write_json(
        tmp_path / "prompt-a.json",
        {"request": {"prompt_token_ids": [1, 2, 3]}, "token_result": {}},
    )
    prompt_b = _write_json(
        tmp_path / "prompt-b.json",
        {"request": {"prompt_token_ids": [1, 2, 3]}, "token_result": {}},
    )
    prepared = _write_json(tmp_path / "manifest.json", {"schema": "manifest"})
    profile = _write_json(tmp_path / "profile.json", {"schema": "profile"})
    audit = _audit(tmp_path / "audit.json")

    plan = build_plan(
        prompt_paths=[prompt_a, prompt_b],
        variants=[
            Variant(
                name="top5",
                prepared=prepared,
                launch_profile=profile,
                launch_audit=audit,
                env={},
            )
        ],
        output_dir=tmp_path / "runs",
        result_prefix="matrix",
        max_new_tokens=1,
        runner="metal/largerlm-runner",
        max_prompt_tokens=64,
    )

    assert plan["schema"] == "largerlm.multi_prompt_replay_plan.v1"
    assert plan["prompt_count"] == 1
    assert plan["variant_count"] == 1
    assert plan["task_count"] == 1
    assert plan["all_tasks_ready"] is True
    task = plan["tasks"][0]
    assert task["ready"] is True
    assert "--require-locked-launch-profile" in task["argv"]
    assert task["result_path"].endswith("-top5-prompt-a-max1-generate-tokenids.json")


def test_build_plan_blocks_prompt_over_limit(tmp_path: Path) -> None:
    prompt = _write_json(
        tmp_path / "prompt.json",
        {"request": {"prompt_token_ids": [1, 2, 3, 4]}, "token_result": {}},
    )
    prepared = _write_json(tmp_path / "manifest.json", {"schema": "manifest"})
    profile = _write_json(tmp_path / "profile.json", {"schema": "profile"})
    audit = _audit(tmp_path / "audit.json")

    plan = build_plan(
        prompt_paths=[prompt],
        variants=[
            Variant(
                name="top6",
                prepared=prepared,
                launch_profile=profile,
                launch_audit=audit,
                env={},
            )
        ],
        output_dir=tmp_path / "runs",
        result_prefix="matrix",
        max_new_tokens=1,
        runner="metal/largerlm-runner",
        max_prompt_tokens=3,
    )

    assert plan["all_tasks_ready"] is False
    assert plan["tasks"][0]["blocked_reasons"] == ["prompt_exceeds_max_prompt_tokens"]


def test_build_plan_reuses_existing_result_for_matching_variant_prompt(
    tmp_path: Path,
) -> None:
    prompt = _write_json(
        tmp_path / "prompt.json",
        {"request": {"prompt_token_ids": [7, 8]}, "token_result": {}},
    )
    existing = _write_json(
        tmp_path / "existing.json",
        {
            "request": {"prompt_token_ids": [7, 8]},
            "token_result": {"generated_token_ids": [9]},
        },
    )
    prepared = _write_json(tmp_path / "manifest.json", {"schema": "manifest"})
    profile = _write_json(tmp_path / "profile.json", {"schema": "profile"})
    audit = _audit(tmp_path / "audit.json")

    plan = build_plan(
        prompt_paths=[prompt],
        variants=[
            Variant(
                name="top5",
                prepared=prepared,
                launch_profile=profile,
                launch_audit=audit,
                env={},
            )
        ],
        existing_results=[("top5", existing)],
        output_dir=tmp_path / "runs",
        result_prefix="matrix",
        max_new_tokens=1,
        runner="metal/largerlm-runner",
        max_prompt_tokens=64,
    )

    task = plan["tasks"][0]
    assert task["result_path"] == str(existing)
    assert task["result_exists"] is True
    assert task["planned_result_path"] != str(existing)
    assert task["existing_result"]["generated_token_ids"] == [9]


def test_multi_prompt_bakeoff_reports_incomplete_when_results_missing(
    tmp_path: Path,
) -> None:
    plan = {
        "schema": "largerlm.multi_prompt_replay_plan.v1",
        "prompts": [{"label": "p0"}, {"label": "p1"}],
        "tasks": [
            {
                "prompt_label": "p0",
                "variant": "top5",
                "result_path": str(_write_json(tmp_path / "p0-top5.json", {})),
            },
            {
                "prompt_label": "p0",
                "variant": "top6",
                "result_path": str(_write_json(tmp_path / "p0-top6.json", {})),
            },
            {
                "prompt_label": "p1",
                "variant": "top5",
                "result_path": str(tmp_path / "missing-top5.json"),
            },
            {
                "prompt_label": "p1",
                "variant": "top6",
                "result_path": str(tmp_path / "missing-top6.json"),
            },
        ],
    }

    payload = build_multi_prompt_bakeoff(
        plan,
        baseline_variant="top5",
        candidate_variants=["top6"],
    )

    assert payload["overall_decision"] == "incomplete"
    assert payload["completed_prompt_count"] == 1
    assert payload["candidate_stats"]["top6"]["missing"] == 1


def test_multi_prompt_bakeoff_retains_baseline_unless_candidate_wins_all(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = {
        "schema": "largerlm.multi_prompt_replay_plan.v1",
        "prompts": [{"label": "p0"}, {"label": "p1"}],
        "tasks": [
            {
                "prompt_label": "p0",
                "variant": "top5",
                "result_path": str(_write_json(tmp_path / "p0-top5.json", {})),
            },
            {
                "prompt_label": "p0",
                "variant": "top6",
                "result_path": str(_write_json(tmp_path / "p0-top6.json", {})),
            },
            {
                "prompt_label": "p1",
                "variant": "top5",
                "result_path": str(_write_json(tmp_path / "p1-top5.json", {})),
            },
            {
                "prompt_label": "p1",
                "variant": "top6",
                "result_path": str(_write_json(tmp_path / "p1-top6.json", {})),
            },
        ],
    }

    def fake_bakeoff(baseline_path, candidate_paths, **kwargs):
        candidate_path = str(candidate_paths[0])
        if "p0-top6" in candidate_path:
            return {
                "selected": {"role": "candidate", "path": candidate_path},
                "winner": {"path": candidate_path},
                "baseline_retained": False,
                "candidates": [
                    {
                        "path": candidate_path,
                        "decision": "prefer_candidate",
                    }
                ],
            }
        return {
            "selected": {"role": "baseline", "path": str(baseline_path)},
            "winner": None,
            "baseline_retained": True,
            "candidates": [
                {
                    "path": candidate_path,
                    "decision": "tie",
                }
            ],
        }

    monkeypatch.setattr(
        "scripts.multi_prompt_bakeoff.result_bakeoff_files",
        fake_bakeoff,
    )

    payload = build_multi_prompt_bakeoff(
        plan,
        baseline_variant="top5",
        candidate_variants=["top6"],
    )

    assert payload["completed_prompt_count"] == 2
    assert payload["candidate_stats"]["top6"]["wins"] == 1
    assert payload["candidate_stats"]["top6"]["ties"] == 1
    assert payload["overall_decision"] == "retain_baseline"
