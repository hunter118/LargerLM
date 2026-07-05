from __future__ import annotations

import json
from pathlib import Path

import pytest

from largerlm.routed_read import (
    RoutedExpertReadError,
    combine_prefill_guard_flags,
    estimate_routed_prefill_chunk_frontier,
    estimate_routed_stage_temp,
    suggest_decode_routed_read_guard_flags,
    suggest_routed_read_guard_flags,
    suggest_routed_stage_temp_guard_flags,
)


def test_suggest_decode_routed_read_guard_flags_with_ssd_seconds() -> None:
    suggested = suggest_decode_routed_read_guard_flags(
        read_bytes_per_token=384,
        ssd_read_gib_per_second=16.0,
        source="unit-test",
    )

    assert suggested is not None
    assert suggested["source"] == "unit-test"
    assert suggested["headroom_factor"] == 1.05
    assert suggested["decode_read_bytes_per_token"] == 384
    assert suggested["decode_max_routed_read_gib_per_token"] == pytest.approx(
        384 / 1024**3 * 1.05
    )
    assert suggested["prefill_ssd_read_gib_per_second"] == 16.0
    assert suggested["decode_read_seconds_per_token"] == pytest.approx(
        384 / (16 * 1024**3)
    )
    assert suggested["decode_max_routed_read_seconds_per_token"] == pytest.approx(
        384 / (16 * 1024**3) * 1.05
    )
    argv = suggested["argv"]
    assert isinstance(argv, tuple)
    assert argv[0] == "--decode-max-routed-read-gib-per-token"
    assert float(argv[1]) == pytest.approx(384 / 1024**3 * 1.05, rel=1e-5)
    assert argv[2] == "--prefill-ssd-read-gib-s"
    assert argv[3] == "16"
    assert argv[4] == "--decode-max-routed-read-seconds-per-token"
    assert float(argv[5]) == pytest.approx(
        384 / (16 * 1024**3) * 1.05,
        rel=1e-5,
    )


def test_suggest_decode_routed_read_guard_flags_returns_none_without_read() -> None:
    assert (
        suggest_decode_routed_read_guard_flags(
            read_bytes_per_token=0,
            ssd_read_gib_per_second=16.0,
        )
        is None
    )
    assert (
        suggest_decode_routed_read_guard_flags(
            read_bytes_per_token=None,
            ssd_read_gib_per_second=16.0,
        )
        is None
    )


def test_suggest_decode_routed_read_guard_flags_rejects_bad_headroom() -> None:
    with pytest.raises(RoutedExpertReadError, match="headroom_factor must be finite"):
        suggest_decode_routed_read_guard_flags(
            read_bytes_per_token=384,
            headroom_factor=float("inf"),
        )


def test_suggest_routed_read_guard_flags_with_ssd_seconds() -> None:
    suggested = suggest_routed_read_guard_flags(
        prompt_chunk_tokens=64,
        planned_read_bytes=1024**3,
        read_amplification=1.2,
        ssd_read_gib_per_second=10.0,
        source="unit-test",
    )

    assert suggested is not None
    assert suggested["source"] == "unit-test"
    assert suggested["headroom_factor"] == 1.05
    assert suggested["prefill_prompt_chunk_tokens"] == 64
    assert suggested["prefill_max_routed_read_amplification"] == pytest.approx(1.26)
    assert suggested["prefill_max_routed_read_gib"] == pytest.approx(1.05)
    assert suggested["prefill_ssd_read_gib_per_second"] == 10.0
    assert suggested["planned_routed_read_seconds"] == pytest.approx(0.1)
    assert suggested["prefill_max_routed_read_seconds"] == pytest.approx(0.105)
    assert suggested["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "64",
        "--prefill-max-routed-read-amplification",
        "1.26",
        "--prefill-max-routed-read-gib",
        "1.05",
        "--prefill-ssd-read-gib-s",
        "10",
        "--prefill-max-routed-read-seconds",
        "0.105",
    )


def test_suggest_routed_read_guard_flags_returns_none_without_read() -> None:
    assert (
        suggest_routed_read_guard_flags(
            prompt_chunk_tokens=64,
            planned_read_bytes=0,
            read_amplification=1.0,
        )
        is None
    )
    assert (
        suggest_routed_read_guard_flags(
            prompt_chunk_tokens=64,
            planned_read_bytes=1024,
            read_amplification=float("nan"),
        )
        is None
    )


def test_suggest_routed_read_guard_flags_rejects_bad_headroom() -> None:
    with pytest.raises(RoutedExpertReadError, match="headroom_factor must be positive"):
        suggest_routed_read_guard_flags(
            prompt_chunk_tokens=64,
            planned_read_bytes=1024,
            read_amplification=1.0,
            headroom_factor=0.0,
        )


def test_suggest_routed_stage_temp_guard_flags() -> None:
    suggested = suggest_routed_stage_temp_guard_flags(
        prompt_chunk_tokens=64,
        max_stage_bytes=1024**2,
        max_compact_stage_bytes=2 * 1024**2,
        max_stage_plus_compact_bytes=3 * 1024**2,
        total_stage_plus_compact_bytes=9 * 1024**2,
        source="unit-test",
    )

    assert suggested is not None
    assert suggested["source"] == "unit-test"
    assert suggested["headroom_factor"] == 1.05
    assert suggested["prefill_prompt_chunk_tokens"] == 64
    assert suggested["prefill_max_stage_mib"] == pytest.approx(1.05)
    assert suggested["prefill_max_compact_stage_mib"] == pytest.approx(2.1)
    assert suggested["profile_max_stage_bytes"] == 1024**2
    assert suggested["profile_max_compact_stage_bytes"] == 2 * 1024**2
    assert suggested["profile_max_stage_plus_compact_bytes"] == 3 * 1024**2
    assert suggested["profile_total_stage_plus_compact_bytes"] == 9 * 1024**2
    assert suggested["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "64",
        "--prefill-max-stage-mib",
        "1.05",
        "--prefill-max-compact-stage-mib",
        "2.1",
    )


def test_suggest_routed_stage_temp_guard_flags_returns_none_without_profile() -> None:
    assert (
        suggest_routed_stage_temp_guard_flags(
            prompt_chunk_tokens=64,
            max_stage_bytes=0,
            max_compact_stage_bytes=1024,
        )
        is None
    )
    assert (
        suggest_routed_stage_temp_guard_flags(
            prompt_chunk_tokens=None,
            max_stage_bytes=1024,
            max_compact_stage_bytes=1024,
        )
        is None
    )


def test_combine_prefill_guard_flags_deduplicates_prompt_chunk() -> None:
    routed = suggest_routed_read_guard_flags(
        prompt_chunk_tokens=64,
        planned_read_bytes=1024**3,
        read_amplification=1.2,
        ssd_read_gib_per_second=10.0,
        source="unit-read",
    )
    stage = suggest_routed_stage_temp_guard_flags(
        prompt_chunk_tokens=64,
        max_stage_bytes=1024**2,
        max_compact_stage_bytes=2 * 1024**2,
        source="unit-stage",
    )

    combined = combine_prefill_guard_flags(
        routed_read_flags=routed,
        stage_temp_flags=stage,
        source="unit-combined",
    )

    assert combined is not None
    assert combined["source"] == "unit-combined"
    assert combined["prefill_prompt_chunk_tokens"] == 64
    assert combined["routed_read_guard"] == routed
    assert combined["stage_temp_guard"] == stage
    assert combined["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "64",
        "--prefill-max-routed-read-amplification",
        "1.26",
        "--prefill-max-routed-read-gib",
        "1.05",
        "--prefill-ssd-read-gib-s",
        "10",
        "--prefill-max-routed-read-seconds",
        "0.105",
        "--prefill-max-stage-mib",
        "1.05",
        "--prefill-max-compact-stage-mib",
        "2.1",
    )
    assert combined["argv"].count("--prefill-prompt-chunk-tokens") == 1


def test_combine_prefill_guard_flags_accepts_stage_only() -> None:
    stage = suggest_routed_stage_temp_guard_flags(
        prompt_chunk_tokens=32,
        max_stage_bytes=1024**2,
        max_compact_stage_bytes=1024**2,
        source="unit-stage",
    )

    combined = combine_prefill_guard_flags(
        routed_read_flags=None,
        stage_temp_flags=stage,
        source="unit-combined",
    )

    assert combined is not None
    assert combined["source"] == "unit-combined"
    assert combined["prefill_prompt_chunk_tokens"] == 32
    assert combined["argv"] == (
        "--prefill-prompt-chunk-tokens",
        "32",
        "--prefill-max-stage-mib",
        "1.05",
        "--prefill-max-compact-stage-mib",
        "1.05",
    )
    assert "routed_read_guard" not in combined
    assert combined["stage_temp_guard"] == stage


def test_estimate_routed_stage_temp_uses_actual_prompt_chunks(
    tmp_path: Path,
) -> None:
    layout = tmp_path / "experts.json"
    layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 0,
                        "num_experts": 8,
                        "expert_slot_bytes": 10,
                    },
                    {
                        "layer": 1,
                        "num_experts": 3,
                        "expert_slot_bytes": 20,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    estimate = estimate_routed_stage_temp(
        expert_layout_path=layout,
        prompt_token_count=5,
        prompt_chunk_tokens=3,
        top_k=2,
        stage_align_bytes=4,
    )

    assert estimate.prompt_chunk_tokens == 3
    assert estimate.top_k == 2
    assert estimate.chunks_per_prompt == 2
    assert estimate.layers == 2
    assert estimate.stage_align_bytes == 4
    assert estimate.static_capacity_per_expert is None
    assert estimate.allow_static_capacity_overflow is False
    assert estimate.max_static_capacity_per_expert == 0
    assert estimate.static_capacity_strict_overflow_safe is True
    assert estimate.max_unique_experts_per_layer == 6
    assert estimate.max_stage_bytes == 84
    assert estimate.max_compact_stage_bytes == 60
    assert estimate.max_stage_plus_compact_bytes == 144
    assert estimate.max_static_capacity_binary_bytes == 0
    assert estimate.max_static_capacity_overflow_records == 0
    assert estimate.max_stage_plus_compact_plus_static_bytes == 144
    assert estimate.max_chunk_stage_bytes == 156
    assert estimate.max_chunk_compact_stage_bytes == 120
    assert estimate.max_chunk_stage_plus_compact_bytes == 276
    assert estimate.max_chunk_static_capacity_binary_bytes == 0
    assert estimate.max_chunk_stage_plus_compact_plus_static_bytes == 276
    assert estimate.total_stage_bytes == 284
    assert estimate.total_compact_stage_bytes == 220
    assert estimate.total_stage_plus_compact_bytes == 504
    assert estimate.total_static_capacity_binary_bytes == 0
    assert estimate.total_static_capacity_overflow_records == 0
    assert estimate.total_stage_plus_compact_plus_static_bytes == 504

    static_estimate = estimate_routed_stage_temp(
        expert_layout_path=layout,
        prompt_token_count=5,
        prompt_chunk_tokens=3,
        top_k=2,
        stage_align_bytes=4,
        static_capacity_per_expert="auto",
    )

    assert static_estimate.static_capacity_per_expert == "auto"
    assert static_estimate.allow_static_capacity_overflow is False
    assert static_estimate.max_static_capacity_per_expert == 3
    assert static_estimate.static_capacity_strict_overflow_safe is True
    assert static_estimate.max_static_capacity_binary_bytes == 280
    assert static_estimate.max_chunk_static_capacity_binary_bytes == 440
    assert static_estimate.total_static_capacity_binary_bytes == 716
    assert static_estimate.max_static_capacity_overflow_records == 0
    assert static_estimate.total_static_capacity_overflow_records == 0
    assert static_estimate.max_stage_plus_compact_plus_static_bytes == 424
    assert static_estimate.max_chunk_stage_plus_compact_plus_static_bytes == 716
    assert static_estimate.total_stage_plus_compact_plus_static_bytes == 1220


def test_estimate_routed_prefill_chunk_frontier_profiles_candidates(
    tmp_path: Path,
) -> None:
    layout = tmp_path / "experts.json"
    layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 0,
                        "num_experts": 8,
                        "expert_slot_bytes": 10,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frontier = estimate_routed_prefill_chunk_frontier(
        expert_layout_path=layout,
        prompt_token_count=8,
        top_k=2,
        stage_align_bytes=4,
        include_chunk_tokens=(3,),
        ssd_read_gib_per_second=10.0,
    )

    assert frontier.prompt_token_count == 8
    assert frontier.top_k == 2
    assert frontier.layers == 1
    assert frontier.stage_align_bytes == 4
    assert frontier.baseline_read_bytes == 80
    assert frontier.saturation_chunk_tokens == 4
    assert [item.prompt_chunk_tokens for item in frontier.candidates] == [
        1,
        2,
        3,
        4,
        8,
    ]
    by_chunk = {item.prompt_chunk_tokens: item for item in frontier.candidates}
    assert by_chunk[3].saturates_all_experts_per_layer is False
    assert by_chunk[4].saturates_all_experts_per_layer is True
    assert by_chunk[4].chunks_per_prompt == 2
    assert by_chunk[4].planned_read_bytes == 160
    assert by_chunk[4].extra_read_bytes == 80
    assert by_chunk[4].read_amplification == 2.0
    assert by_chunk[4].max_layer_planned_read_bytes == 160
    assert by_chunk[4].max_stage_plus_compact_bytes == 192
    assert by_chunk[4].max_chunk_stage_plus_compact_bytes == 192
    assert by_chunk[4].total_stage_plus_compact_bytes == 384
    assert by_chunk[4].max_static_capacity_binary_bytes == 0
    assert by_chunk[4].max_chunk_static_capacity_binary_bytes == 0
    assert by_chunk[4].total_static_capacity_binary_bytes == 0
    assert by_chunk[4].max_stage_plus_compact_plus_static_bytes == 192
    assert by_chunk[4].max_chunk_stage_plus_compact_plus_static_bytes == 192
    assert by_chunk[4].total_stage_plus_compact_plus_static_bytes == 384
    assert by_chunk[4].planned_read_seconds == pytest.approx(160 / (10 * 1024**3))
    assert by_chunk[8].read_amplification == 1.0


def test_estimate_routed_prefill_chunk_frontier_includes_static_route_binary(
    tmp_path: Path,
) -> None:
    layout = tmp_path / "experts.json"
    layout.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "layer": 0,
                        "num_experts": 8,
                        "expert_slot_bytes": 10,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frontier = estimate_routed_prefill_chunk_frontier(
        expert_layout_path=layout,
        prompt_token_count=8,
        top_k=2,
        stage_align_bytes=4,
        include_chunk_tokens=(4,),
        static_capacity_per_expert="auto",
    )

    assert frontier.static_capacity_per_expert == "auto"
    assert frontier.allow_static_capacity_overflow is False
    by_chunk = {item.prompt_chunk_tokens: item for item in frontier.candidates}
    assert by_chunk[4].max_static_capacity_binary_bytes == 456
    assert by_chunk[4].max_chunk_static_capacity_binary_bytes == 456
    assert by_chunk[4].total_static_capacity_binary_bytes == 912
    assert by_chunk[4].max_stage_plus_compact_plus_static_bytes == 648
    assert by_chunk[4].max_chunk_stage_plus_compact_plus_static_bytes == 648
    assert by_chunk[4].total_stage_plus_compact_plus_static_bytes == 1296
