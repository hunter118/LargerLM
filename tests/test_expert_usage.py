from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from largerlm.expert_usage import (
    M5_MAX_128G_SAFE_PIN_BUDGET_BYTES,
    ExpertUsageError,
    build_pin_plan,
    build_usage_profile,
    format_colibri_usage,
)


def test_prefill_hotspots_are_deduped_across_copy_and_range_lists(
    tmp_path: Path,
) -> None:
    source = tmp_path / "result.json"
    row = {
        "chunk_index": 0,
        "layer": 1,
        "tile_index": 2,
        "selected_experts": [3, 7],
    }
    source.write_text(
        json.dumps(
            {
                "token_result": {
                    "prefill_actual_read_time": {
                        "expert_stage_copy_hotspots": [row],
                        "expert_stage_range_hotspots": [row],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    profile = build_usage_profile([source])

    assert profile["source_kinds"] == ["largerlm_prefill_hotspots"]
    assert profile["complete_route_telemetry"] is False
    assert profile["total_selections"] == 2
    assert profile["layers"][0]["experts"] == [
        {"expert": 3, "count": 1, "fraction": 0.5},
        {"expert": 7, "count": 1, "fraction": 0.5},
    ]


def test_colibri_and_jsonl_usage_are_merged(tmp_path: Path) -> None:
    colibri = tmp_path / ".coli_usage"
    events = tmp_path / "routes.jsonl"
    colibri.write_text("1 2 5\n1 3 2\n", encoding="utf-8")
    events.write_text(
        '{"layer": 1, "experts": [2, 4]}\n'
        '{"layer": 2, "selected_experts": [5], "count": 3}\n',
        encoding="utf-8",
    )

    profile = build_usage_profile([colibri, events])

    assert profile["total_selections"] == 12
    assert profile["source_kinds"] == ["colibri_usage", "route_events"]
    assert profile["layers"][0]["experts"][0]["expert"] == 2
    assert profile["layers"][0]["experts"][0]["count"] == 6
    assert format_colibri_usage(profile).startswith("1 2 6\n1 3 2\n1 4 1\n")


def test_router_json_needs_default_layer(tmp_path: Path) -> None:
    source = tmp_path / "router.json"
    source.write_text(json.dumps({"experts": [1, 2]}), encoding="utf-8")

    with pytest.raises(ExpertUsageError, match="default_layer"):
        build_usage_profile([source])

    profile = build_usage_profile([source], default_layer=4)
    assert profile["layers"][0]["layer"] == 4


def test_runtime_generation_expert_routes_are_collected(tmp_path: Path) -> None:
    source = tmp_path / "generated.json"
    source.write_text(
        json.dumps(
            {
                "probe_generate": {
                    "steps": [
                        {
                            "expert_routes": [
                                {"layer": 2, "experts": [1, 4]},
                                {"layer": 3, "experts": [0, 4]},
                            ]
                        },
                        {
                            "expert_routes": [
                                {"layer": 2, "experts": [1, 5]},
                                {"layer": 3, "experts": [0, 4]},
                            ]
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    profile = build_usage_profile([source])

    assert profile["complete_route_telemetry"] is True
    assert profile["total_selections"] == 8
    assert profile["layers"][0]["experts"][0] == {
        "expert": 1,
        "count": 2,
        "fraction": 0.5,
    }


def test_pin_plan_uses_byte_weighted_frequency_and_stays_in_budget(
    tmp_path: Path,
) -> None:
    usage = tmp_path / "usage.txt"
    layout = tmp_path / "layout.json"
    usage.write_text("1 0 10\n1 1 8\n2 0 9\n2 1 7\n", encoding="utf-8")
    layout.write_text(
        json.dumps(
            {
                "layers": [
                    {"layer": 1, "num_experts": 2, "expert_slot_bytes": 100},
                    {"layer": 2, "num_experts": 2, "expert_slot_bytes": 200},
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = build_usage_profile([usage])

    plan = build_pin_plan(
        profile,
        expert_layout_path=layout,
        max_pin_bytes=M5_MAX_128G_SAFE_PIN_BUDGET_BYTES,
        max_experts_per_layer=1,
        target_profile="m5-max-128g-safe",
    )

    assert plan["quality_preserving"] is True
    assert plan["changes_routing"] is False
    assert plan["selected_bytes"] == 300
    assert [(row["layer"], row["expert"]) for row in plan["selected_experts"]] == [
        (1, 0),
        (2, 0),
    ]
    assert plan["covered_profile_selections"] == 19
    assert plan["profile_hit_fraction"] == 19 / 34
    assert plan["memory_envelope"]["requires_runtime_rss_guard"] is True
    assert plan["planner_only"] is False
    assert plan["runtime_consumable"] is True
    assert plan["runtime_cli_flag"] == "--expert-pin-plan"
    assert plan["memory_envelope"]["maximum_expert_resident_gib"] == 10
    assert plan["memory_envelope"]["default_expert_cache_policy"] == "os_page_cache"
    assert plan["memory_envelope"]["experimental_upper_bound"] is False
    assert plan["memory_envelope"]["runtime_live_cap_includes_expert_cache"] is True
    assert plan["memory_envelope"]["metal_recommended_working_set_bound"] is True
    assert (
        plan["memory_envelope"][
            "unallocated_headroom_at_maximum_expert_residency_gib"
        ]
        == 88
    )


def test_cli_writes_m5_max_profile_and_plan(tmp_path: Path) -> None:
    usage = tmp_path / "usage.txt"
    layout = tmp_path / "layout.json"
    profile_path = tmp_path / "profile.json"
    plan_path = tmp_path / "plan.json"
    usage.write_text("3 1 10\n3 2 2\n", encoding="utf-8")
    layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 3,
                        "num_experts": 4,
                        "expert_slot_bytes": 1024,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "expert_usage_plan.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(usage),
            "--expert-layout",
            str(layout),
            "--m5-max-128g-safe",
            "--write-profile",
            str(profile_path),
            "--write-plan",
            str(plan_path),
            "--quiet",
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(profile_path.read_text())["total_selections"] == 12
    plan = json.loads(plan_path.read_text())
    assert plan["target_profile"] == "m5-max-128g-safe"
    assert plan["memory_envelope"]["maximum_expert_resident_gib"] == 10
