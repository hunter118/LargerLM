from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

import largerlm.expert_io as expert_io_module
from largerlm.cli import main as cli_main
from largerlm.expert_io import (
    ExpertIOPlanError,
    plan_batch_expert_io,
    plan_batch_expert_io_tiles,
    plan_expert_io,
    plan_static_expert_capacity,
    stage_batch_experts,
    static_expert_capacity_binary_bytes,
    static_expert_capacity_binary_bytes_for_counts,
    static_expert_capacity_json_bytes,
    static_expert_capacity_payload,
    validate_static_expert_capacity_binary,
    write_static_expert_capacity_binary,
    write_static_expert_capacity_plan,
)


def _write_expert_layout(root: Path, *, slot_bytes: int = 100) -> Path:
    experts = root / "experts"
    experts.mkdir()
    layout = experts / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "quantization": "mlx-affine-int4",
                "group_size": 8,
                "num_layers": 1,
                "num_experts": 8,
                "component_order": ["gate_proj.weight"],
                "layers": [
                    {
                        "layer": 3,
                        "num_experts": 8,
                        "expert_slot_bytes": slot_bytes,
                        "layer_file": "layer_003.bin",
                        "components": [
                            {
                                "name": "gate_proj.weight",
                                "offset": 0,
                                "size": slot_bytes,
                                "dtype": "U32",
                                "shape": [8, 1],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (experts / "layer_003.bin").write_bytes(
        b"".join(bytes([expert]) * slot_bytes for expert in range(8))
    )
    return layout


def _mutate_layout(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_plan_expert_io_coalesces_adjacent_slots(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    plan = plan_expert_io(
        layout,
        layer=3,
        expert_ids=[4, 1, 0, 1],
        merge_gap_bytes=0,
        align_bytes=1,
    )

    assert plan.selected_experts == (0, 1, 4)
    assert plan.requested_bytes == 300
    assert plan.raw_span_bytes == 300
    assert plan.read_bytes == 300
    assert plan.read_amplification == 1.0
    assert [item.experts for item in plan.ranges] == [(0, 1), (4,)]
    assert [(item.offset, item.length) for item in plan.ranges] == [(0, 200), (400, 100)]


def test_plan_expert_io_uses_optional_expert_order(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        layers[0]["expert_order"] = [4, 1, 0, 2, 3, 5, 6, 7]

    _mutate_layout(layout, mutate)

    plan = plan_expert_io(
        layout,
        layer=3,
        expert_ids=[4, 0, 1],
        merge_gap_bytes=0,
        align_bytes=1,
    )

    assert plan.selected_experts == (0, 1, 4)
    assert plan.expert_physical_slots == (2, 1, 3, 4, 0, 5, 6, 7)
    assert plan.raw_range_count == 1
    assert plan.coalesced_range_count == 1
    assert plan.ranges[0].experts == (4, 1, 0)
    assert plan.ranges[0].offset == 0
    assert plan.ranges[0].length == 300


@pytest.mark.parametrize(
    ("expert_order", "message"),
    (
        ([0, 1, 2], "expert_order length"),
        ([0, 1, 2, 3, 4, 5, 6, 6], "must not repeat"),
        ([0, 1, 2, 3, 4, 5, 6, 8], "outside layer expert range"),
        ([0, 1, 2, 3, 4, 5, 6, True], "entries must be integers"),
    ),
)
def test_plan_expert_io_rejects_invalid_expert_order(
    tmp_path: Path,
    expert_order: list[object],
    message: str,
) -> None:
    layout = _write_expert_layout(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        layers[0]["expert_order"] = expert_order

    _mutate_layout(layout, mutate)

    with pytest.raises(ExpertIOPlanError, match=message):
        plan_expert_io(layout, layer=3, expert_ids=[0])


def test_plan_expert_io_rejects_boolean_layer_argument(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    with pytest.raises(ExpertIOPlanError, match="layer must be an integer"):
        plan_expert_io(layout, layer=True, expert_ids=[0])


def test_plan_expert_io_rejects_boolean_layer_id(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        layers[0]["layer"] = True

    _mutate_layout(layout, mutate)

    with pytest.raises(ExpertIOPlanError, match="layer 3 not found"):
        plan_expert_io(layout, layer=3, expert_ids=[0])


def test_plan_expert_io_rejects_boolean_layer_budget_field(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        layers[0]["expert_slot_bytes"] = False

    _mutate_layout(layout, mutate)

    with pytest.raises(
        ExpertIOPlanError,
        match="layout layer expert_slot_bytes must be an integer",
    ):
        plan_expert_io(layout, layer=3, expert_ids=[0])


def test_plan_expert_io_merges_gap_when_requested(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    plan = plan_expert_io(
        layout,
        layer=3,
        expert_ids=[0, 2],
        merge_gap_bytes=100,
        align_bytes=1,
    )

    assert len(plan.ranges) == 1
    assert plan.ranges[0].experts == (0, 2)
    assert plan.ranges[0].length == 300
    assert plan.requested_bytes == 200
    assert plan.read_bytes == 300
    assert plan.waste_bytes == 100
    assert plan.read_amplification == 1.5


def test_plan_expert_io_accounts_for_alignment(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path, slot_bytes=3000)

    plan = plan_expert_io(
        layout,
        layer=3,
        expert_ids=[1],
        merge_gap_bytes=0,
        align_bytes=4096,
    )

    assert plan.ranges[0].offset == 3000
    assert plan.ranges[0].aligned_offset == 0
    assert plan.ranges[0].aligned_length == 8192
    assert plan.requested_bytes == 3000
    assert plan.waste_bytes == 5192


def test_plan_expert_io_reports_alignment_range_coalescing(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path, slot_bytes=3000)

    plan = plan_expert_io(
        layout,
        layer=3,
        expert_ids=[0, 2],
        merge_gap_bytes=0,
        align_bytes=4096,
    )

    assert plan.raw_range_count == 2
    assert plan.coalesced_range_count == 1
    assert plan.coalesced_range_savings == 1
    assert plan.read_bytes == 12288
    assert plan.waste_bytes == 6288


def test_plan_expert_io_rejects_out_of_range_expert(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    with pytest.raises(ExpertIOPlanError, match="outside layer expert range"):
        plan_expert_io(layout, layer=3, expert_ids=[8])


def test_plan_expert_io_rejects_bool_expert_id(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    with pytest.raises(ExpertIOPlanError, match="expert ids"):
        plan_expert_io(layout, layer=3, expert_ids=[True])


def test_plan_expert_io_rejects_float_expert_id(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    with pytest.raises(ExpertIOPlanError, match="expert ids"):
        plan_expert_io(layout, layer=3, expert_ids=[1.5])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"merge_gap_bytes": True}, "merge_gap_bytes must be an integer"),
        ({"merge_gap_bytes": 1.5}, "merge_gap_bytes must be an integer"),
        ({"align_bytes": False}, "align_bytes must be an integer"),
        ({"align_bytes": 1.5}, "align_bytes must be an integer"),
        ({"align_bytes": 0}, "align_bytes must be positive"),
    ),
)
def test_plan_expert_io_rejects_non_integer_read_advice_controls(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    layout = _write_expert_layout(tmp_path)

    with pytest.raises(ExpertIOPlanError, match=message):
        plan_expert_io(layout, layer=3, expert_ids=[0], **kwargs)


def test_plan_expert_io_rejects_escaped_layer_file(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    payload = json.loads(layout.read_text(encoding="utf-8"))
    payload["layers"][0]["layer_file"] = "../layer_003.bin"
    layout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExpertIOPlanError, match="layer_file"):
        plan_expert_io(layout, layer=3, expert_ids=[0])


def test_plan_expert_io_cli_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    layout = _write_expert_layout(tmp_path)

    status = cli_main(
        [
            "plan-expert-io",
            str(layout),
            "--layer",
            "3",
            "--experts",
            "0,1,4",
            "--align-kib",
            "0.0009765625",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selected_experts"] == [0, 1, 4]
    assert payload["read_bytes"] == 300
    assert len(payload["ranges"]) == 2


def _write_router_jsons(root: Path) -> Path:
    router_dir = root / "router_json"
    router_dir.mkdir()
    routes = [
        {"experts": [0, 2], "weights": [0.7, 0.3]},
        {"experts": [2, 4], "weights": [0.6, 0.4]},
        {"experts": [1], "weights": [1.0]},
    ]
    for index, payload in enumerate(routes):
        (router_dir / f"token_{index:06d}.router.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    return router_dir


def test_plan_batch_expert_io_aggregates_router_json(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)

    plan = plan_batch_expert_io(
        layout,
        layer=3,
        router_json_dir=router_dir,
        merge_gap_bytes=0,
        align_bytes=1,
        ssd_read_gib_per_second=10.0,
    )

    assert plan.batch_tokens == 3
    assert plan.selected_experts == (0, 1, 2, 4)
    assert plan.total_assignments == 5
    assert plan.serial_read_bytes == 500
    assert plan.unique_requested_bytes == 400
    assert plan.planned_read_bytes == 400
    assert plan.coalesced_savings_bytes == 100
    assert plan.raw_range_count == 2
    assert plan.coalesced_range_count == 2
    assert plan.coalesced_range_savings == 0
    assert plan.assignment_read_amplification == 0.8
    assert plan.unique_read_amplification == 1.0
    assert plan.ssd_read_gib_per_second == 10.0
    assert plan.planned_read_seconds == pytest.approx(400 / (10.0 * 1024**3))
    assert [item.expert for item in plan.expert_tokens] == [0, 1, 2, 4]
    assert plan.expert_tokens[2].tokens == (0, 1)
    assert plan.expert_tokens[2].weights == (0.3, 0.6)
    assert [item.experts for item in plan.io_plan.ranges] == [(0, 1, 2), (4,)]


def test_plan_batch_expert_io_tiles_selected_experts_under_stage_caps(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)

    tiling = plan_batch_expert_io_tiles(
        layout,
        layer=3,
        router_json_dir=router_dir,
        merge_gap_bytes=0,
        align_bytes=1,
        max_stage_mib=250 / 1024**2,
        max_compact_stage_mib=250 / 1024**2,
        ssd_read_gib_per_second=10.0,
    )

    assert tiling.batch_plan.selected_experts == (0, 1, 2, 4)
    assert tiling.tile_count == 2
    assert tiling.total_tile_assignments == tiling.batch_plan.total_assignments
    assert tiling.total_tile_planned_read_bytes == 400
    assert tiling.max_tile_planned_read_bytes == 200
    assert tiling.max_tile_compact_stage_bytes == 200
    assert [tile.selected_experts for tile in tiling.tiles] == [(0, 1), (2, 4)]
    assert [tile.active_token_count for tile in tiling.tiles] == [2, 2]
    assert [tile.total_assignments for tile in tiling.tiles] == [2, 3]
    assert [
        tile.planned_read_seconds for tile in tiling.tiles
    ] == pytest.approx([200 / (10.0 * 1024**3)] * 2)


def test_plan_batch_expert_io_tiles_rejects_single_expert_over_cap(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)

    with pytest.raises(ExpertIOPlanError, match="cannot fit one tile"):
        plan_batch_expert_io_tiles(
            layout,
            layer=3,
            router_json_dir=router_dir,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=50 / 1024**2,
            max_compact_stage_mib=50 / 1024**2,
        )


def test_plan_static_expert_capacity_reports_overflow(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(
        layout,
        layer=3,
        router_json_dir=router_dir,
        merge_gap_bytes=0,
        align_bytes=1,
    )

    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=1)

    assert plan.batch_tokens == 3
    assert plan.selected_experts == (0, 1, 2, 4)
    assert plan.total_assignments == 5
    assert plan.total_capacity_slots == 4
    assert plan.used_slots == 4
    assert plan.utilization == 1.0
    assert plan.overflow_assignments == 1
    assert plan.max_tokens_per_expert == 2
    assert plan.requires_overflow_path
    assert [(item.expert, item.used_slots, item.overflow_assignments) for item in plan.usages] == [
        (0, 1, 0),
        (1, 1, 0),
        (2, 1, 1),
        (4, 1, 0),
    ]
    assert [(item.expert, item.slot, item.token_index) for item in plan.slots] == [
        (0, 0, 0),
        (1, 0, 2),
        (2, 0, 0),
        (4, 0, 1),
    ]
    assert [(item.expert, item.overflow_index, item.token_index) for item in plan.overflow] == [
        (2, 0, 1),
    ]


def test_plan_static_expert_capacity_fits_when_capacity_is_sufficient(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)

    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=2)

    assert plan.total_capacity_slots == 8
    assert plan.used_slots == 5
    assert plan.utilization == pytest.approx(0.625)
    assert plan.overflow_assignments == 0
    assert not plan.requires_overflow_path
    assert [(item.expert, item.slot, item.token_index) for item in plan.slots if item.expert == 2] == [
        (2, 0, 0),
        (2, 1, 1),
    ]


def test_plan_static_expert_capacity_rejects_invalid_capacity(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)

    with pytest.raises(ExpertIOPlanError, match="capacity_per_expert must be positive"):
        plan_static_expert_capacity(batch_plan, capacity_per_expert=0)


@pytest.mark.parametrize(
    ("capacity_per_expert", "message"),
    (
        (True, "capacity_per_expert must be an integer"),
        (1.5, "capacity_per_expert must be an integer"),
    ),
)
def test_plan_static_expert_capacity_rejects_non_integer_capacity(
    tmp_path: Path,
    capacity_per_expert: object,
    message: str,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)

    with pytest.raises(ExpertIOPlanError, match=message):
        plan_static_expert_capacity(
            batch_plan,
            capacity_per_expert=capacity_per_expert,
        )


def test_static_expert_capacity_payload_has_fixed_slots(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=2)

    payload = static_expert_capacity_payload(plan)

    assert payload["format"] == "largerlm.static_expert_capacity.v1"
    assert payload["requires_overflow_path"] is False
    assert payload["capacity_per_expert"] == 2
    expert_two = payload["experts"][2]
    assert expert_two["expert"] == 2
    assert expert_two["slots"] == [
        {"active": True, "slot": 0, "token_index": 0, "weight": 0.3},
        {"active": True, "slot": 1, "token_index": 1, "weight": 0.6},
    ]
    expert_zero = payload["experts"][0]
    assert expert_zero["slots"][1] == {
        "active": False,
        "slot": 1,
        "token_index": None,
        "weight": 0.0,
    }


def test_write_static_expert_capacity_plan_rejects_overflow_by_default(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=1)

    with pytest.raises(ExpertIOPlanError, match="overflow assignments"):
        write_static_expert_capacity_plan(plan, tmp_path / "static_routes.json")


def test_write_static_expert_capacity_plan_can_include_overflow(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=1)
    output = tmp_path / "static_routes.json"

    result = write_static_expert_capacity_plan(plan, output, allow_overflow=True)

    assert result == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["overflow_assignments"] == 1
    assert payload["experts"][2]["overflow"] == [
        {"overflow_index": 0, "token_index": 1, "weight": 0.6},
    ]


def test_write_static_expert_capacity_plan_cleans_up_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=2)
    output = tmp_path / "static_routes.json"

    def fail_json(path: Path, payload: object) -> None:
        raise OSError("json exploded")

    monkeypatch.setattr(expert_io_module, "_write_json_atomic", fail_json)

    with pytest.raises(ExpertIOPlanError, match="failed to write static capacity plan"):
        write_static_expert_capacity_plan(plan, output)

    assert not output.exists()


def test_write_static_expert_capacity_binary_uses_fixed_slot_table(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=2)
    output = tmp_path / "static_capacity.bin"

    report = write_static_expert_capacity_binary(plan, output)

    raw = output.read_bytes()
    header = struct.unpack_from("<8sIIIIIIII", raw, 0)
    assert header == (b"LLMSCAP1", 1, 3, 4, 2, 5, 5, 0, 0)
    assert static_expert_capacity_binary_bytes(plan) == len(raw)
    assert (
        static_expert_capacity_binary_bytes_for_counts(
            expert_count=4,
            capacity_per_expert=2,
            overflow_records=0,
        )
        == len(raw)
    )
    assert static_expert_capacity_json_bytes(plan) == len(
        json.dumps(
            static_expert_capacity_payload(plan),
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
    )
    assert report.bytes_written == len(raw)
    assert report.slot_records == 8
    assert report.overflow_records == 0
    validation = validate_static_expert_capacity_binary(output, expected_plan=plan)
    assert validation.bytes_read == len(raw)
    assert validation.version == 1
    assert validation.batch_tokens == 3
    assert validation.expert_count == 4
    assert validation.capacity_per_expert == 2
    assert validation.total_assignments == 5
    assert validation.used_slots == 5
    assert validation.slot_records == 8
    assert validation.active_slot_records == 5
    assert validation.inactive_slot_records == 3
    offset = struct.calcsize("<8sIIIIIIII")
    assert struct.unpack_from("<4I", raw, offset) == (0, 1, 2, 4)
    offset += 4 * 4
    slots = [
        struct.unpack_from("<IfI", raw, offset + index * struct.calcsize("<IfI"))
        for index in range(8)
    ]
    assert slots[0] == (0, pytest.approx(0.7), 1)
    assert slots[1] == (0xFFFFFFFF, pytest.approx(0.0), 0)
    assert slots[4] == (0, pytest.approx(0.3), 1)
    assert slots[5] == (1, pytest.approx(0.6), 1)


def test_write_static_expert_capacity_binary_can_include_overflow(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=1)
    output = tmp_path / "static_capacity.bin"

    report = write_static_expert_capacity_binary(plan, output, allow_overflow=True)

    raw = output.read_bytes()
    assert report.overflow_records == 1
    validation = validate_static_expert_capacity_binary(output, expected_plan=plan)
    assert validation.overflow_records == 1
    assert validation.total_assignments == 5
    assert validation.used_slots == 4
    offset = struct.calcsize("<8sIIIIIIII") + 4 * 4 + 4 * struct.calcsize("<IfI")
    assert struct.unpack_from("<IIIf", raw, offset) == (2, 0, 1, pytest.approx(0.6))


def test_validate_static_expert_capacity_binary_rejects_bad_size(
    tmp_path: Path,
) -> None:
    output = tmp_path / "static_capacity.bin"
    output.write_bytes(
        struct.pack("<8sIIIIIIII", b"LLMSCAP1", 1, 3, 1, 2, 0, 0, 0, 0)
    )

    with pytest.raises(ExpertIOPlanError, match="expected"):
        validate_static_expert_capacity_binary(output)


def test_validate_static_expert_capacity_binary_rejects_bad_slot_record(
    tmp_path: Path,
) -> None:
    output = tmp_path / "static_capacity.bin"
    output.write_bytes(
        b"".join(
            (
                struct.pack("<8sIIIIIIII", b"LLMSCAP1", 1, 3, 1, 1, 1, 1, 0, 0),
                struct.pack("<I", 0),
                struct.pack("<IfI", 99, 1.0, 1),
            )
        )
    )

    with pytest.raises(ExpertIOPlanError, match="slot token_index"):
        validate_static_expert_capacity_binary(output)


def test_validate_static_expert_capacity_binary_rejects_plan_mismatch(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=2)
    output = tmp_path / "static_capacity.bin"
    write_static_expert_capacity_binary(plan, output)
    different_plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=1)

    with pytest.raises(ExpertIOPlanError, match="capacity_per_expert mismatches"):
        validate_static_expert_capacity_binary(output, expected_plan=different_plan)


def test_validate_static_capacity_bin_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=2)
    output = tmp_path / "static_capacity.bin"
    write_static_expert_capacity_binary(plan, output)

    status = cli_main(
        [
            "validate-static-capacity-bin",
            str(output),
            "--expect-batch-tokens",
            "3",
            "--expect-expert-count",
            "4",
            "--expect-capacity-per-expert",
            "2",
            "--expect-total-assignments",
            "5",
            "--expect-used-slots",
            "5",
            "--expect-overflow-records",
            "0",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == str(output)
    assert payload["batch_tokens"] == 3
    assert payload["expert_count"] == 4
    assert payload["active_slot_records"] == 5


def test_validate_static_capacity_bin_cli_rejects_expectation_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=2)
    output = tmp_path / "static_capacity.bin"
    write_static_expert_capacity_binary(plan, output)

    status = cli_main(
        [
            "validate-static-capacity-bin",
            str(output),
            "--expect-used-slots",
            "4",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "used_slots=5 does not match expected 4" in captured.err


def test_write_static_expert_capacity_binary_cleans_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    batch_plan = plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)
    plan = plan_static_expert_capacity(batch_plan, capacity_per_expert=2)
    output = tmp_path / "static_capacity.bin"

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("replace exploded")

    monkeypatch.setattr(type(output), "replace", fail_replace)

    with pytest.raises(ExpertIOPlanError, match="failed to write static capacity binary"):
        write_static_expert_capacity_binary(plan, output)

    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()


def test_plan_batch_expert_io_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)

    status = cli_main(
        [
            "plan-batch-expert-io",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--align-kib",
            "0.0009765625",
            "--ssd-read-gib-s",
            "10",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_tokens"] == 3
    assert payload["selected_experts"] == [0, 1, 2, 4]
    assert payload["total_assignments"] == 5
    assert payload["planned_read_bytes"] == 400
    assert payload["ssd_read_gib_per_second"] == 10.0
    assert payload["planned_read_seconds"] == pytest.approx(400 / (10.0 * 1024**3))
    assert payload["raw_range_count"] == 2
    assert payload["coalesced_range_count"] == 2
    assert payload["expert_tokens"][2]["tokens"] == [0, 1]


def test_plan_batch_expert_io_cli_json_with_static_capacity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)

    status = cli_main(
        [
            "plan-batch-expert-io",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--align-kib",
            "0.0009765625",
            "--static-capacity-per-expert",
            "1",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_expert_io_plan"]["planned_read_bytes"] == 400
    assert payload["static_capacity_plan"]["capacity_per_expert"] == 1
    assert payload["static_capacity_plan"]["overflow_assignments"] == 1
    assert payload["static_capacity_plan"]["requires_overflow_path"]


def test_plan_batch_expert_io_cli_json_with_tiling_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)

    status = cli_main(
        [
            "plan-batch-expert-io",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--align-kib",
            "0.0009765625",
            "--tile-max-stage-mib",
            str(250 / 1024**2),
            "--tile-max-compact-stage-mib",
            str(250 / 1024**2),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["batch_expert_io_plan"]["planned_read_bytes"] == 400
    tiling = payload["batch_expert_io_tiling_plan"]
    assert tiling["tile_count"] == 2
    assert tiling["total_tile_assignments"] == 5
    assert tiling["max_tile_planned_read_bytes"] == 200
    assert [tile["selected_experts"] for tile in tiling["tiles"]] == [
        [0, 1],
        [2, 4],
    ]


def test_plan_batch_expert_io_cli_rejects_partial_tiling_args(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)

    status = cli_main(
        [
            "plan-batch-expert-io",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--tile-max-stage-mib",
            "1",
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "--tile-max-stage-mib and --tile-max-compact-stage-mib" in captured.err


def test_plan_batch_expert_io_cli_writes_static_capacity_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    output = tmp_path / "static_capacity.json"

    status = cli_main(
        [
            "plan-batch-expert-io",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--static-capacity-per-expert",
            "2",
            "--static-capacity-output-json",
            str(output),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["static_capacity_plan"]["overflow_assignments"] == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["capacity_per_expert"] == 2
    assert artifact["experts"][0]["slots"][1]["active"] is False


def test_plan_batch_expert_io_cli_writes_static_capacity_binary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    output = tmp_path / "static_capacity.bin"

    status = cli_main(
        [
            "plan-batch-expert-io",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--static-capacity-per-expert",
            "2",
            "--static-capacity-output-bin",
            str(output),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["static_capacity_plan"]["overflow_assignments"] == 0
    assert output.exists()
    assert struct.unpack_from("<8sI", output.read_bytes(), 0) == (b"LLMSCAP1", 1)


def test_plan_batch_expert_io_cli_rejects_static_capacity_output_overflow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    output = tmp_path / "static_capacity.json"

    status = cli_main(
        [
            "plan-batch-expert-io",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--static-capacity-per-expert",
            "1",
            "--static-capacity-output-json",
            str(output),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "overflow assignments" in captured.err
    assert not output.exists()


def test_plan_batch_expert_io_cli_can_write_static_capacity_json_with_overflow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    output = tmp_path / "static_capacity.json"

    status = cli_main(
        [
            "plan-batch-expert-io",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--static-capacity-per-expert",
            "1",
            "--static-capacity-output-json",
            str(output),
            "--allow-static-capacity-overflow",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["static_capacity_plan"]["overflow_assignments"] == 1
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["requires_overflow_path"]
    assert artifact["experts"][2]["overflow"][0]["token_index"] == 1


def test_plan_batch_expert_io_rejects_repeated_expert(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = tmp_path / "router_json"
    router_dir.mkdir()
    (router_dir / "token_000000.router.json").write_text(
        json.dumps({"experts": [1, 1], "weights": [0.5, 0.5]}),
        encoding="utf-8",
    )

    with pytest.raises(ExpertIOPlanError, match="repeats expert"):
        plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)


def test_plan_batch_expert_io_rejects_bool_expert_id(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = tmp_path / "router_json"
    router_dir.mkdir()
    (router_dir / "token_000000.router.json").write_text(
        json.dumps({"experts": [True], "weights": [1.0]}),
        encoding="utf-8",
    )

    with pytest.raises(ExpertIOPlanError, match="non-integer expert"):
        plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)


def test_plan_batch_expert_io_rejects_non_finite_weight(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = tmp_path / "router_json"
    router_dir.mkdir()
    (router_dir / "token_000000.router.json").write_text(
        '{"experts": [1], "weights": [NaN]}',
        encoding="utf-8",
    )

    with pytest.raises(ExpertIOPlanError, match="non-finite weight"):
        plan_batch_expert_io(layout, layer=3, router_json_dir=router_dir)


def test_stage_batch_experts_copies_coalesced_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"
    captured_ranges: list[tuple[int, int]] = []
    pread_calls: list[tuple[int, int]] = []

    def fake_advise(fd: int, ranges: tuple[object, ...]):
        assert fd >= 0
        captured_ranges.extend(
            (item.aligned_offset, item.aligned_length) for item in ranges
        )
        return expert_io_module.ReadAdviceStats(
            supported=True,
            attempted_ranges=len(ranges),
            calls=len(ranges),
            advised_bytes=sum(item.aligned_length for item in ranges),
            error=None,
        )

    real_preadv = os.preadv

    def tracked_preadv(fd: int, buffers, offset: int, *args) -> int:
        pread_calls.append((sum(len(buffer) for buffer in buffers), offset))
        return real_preadv(fd, buffers, offset, *args)

    monkeypatch.setattr(expert_io_module, "_advise_read_ranges", fake_advise)
    monkeypatch.setattr(expert_io_module.os, "preadv", tracked_preadv)

    result = stage_batch_experts(
        layout,
        layer=3,
        router_json_dir=router_dir,
        stage_file_path=stage_file,
        merge_gap_bytes=0,
        align_bytes=1,
        max_stage_mib=1,
        copy_chunk_mib=0.00005,
        ssd_read_gib_per_second=10.0,
        max_read_seconds=1.0,
    )

    assert result.selected_experts == (0, 1, 2, 4)
    assert result.planned_read_bytes == 400
    assert result.staged_bytes == 400
    assert result.copy_chunk_bytes == 52
    assert pread_calls == [
        (52, 0),
        (52, 52),
        (52, 104),
        (52, 156),
        (52, 208),
        (40, 260),
        (52, 400),
        (48, 452),
    ]
    assert all(size <= result.copy_chunk_bytes for size, _ in pread_calls)
    assert result.read_advice.supported is True
    assert result.read_advice.attempted_ranges == 2
    assert result.read_advice.calls == 2
    assert result.read_advice.advised_bytes == 400
    assert captured_ranges == [(0, 300), (400, 100)]
    assert result.io_summary.serial_read_bytes == 500
    assert result.io_summary.unique_requested_bytes == 400
    assert result.io_summary.coalesced_savings_bytes == 100
    assert result.io_summary.stage_budget_utilization == pytest.approx(
        400 / 1024**2
    )
    assert result.io_summary.assignment_read_amplification == pytest.approx(0.8)
    assert result.io_summary.unique_read_amplification == pytest.approx(1.0)
    assert result.io_summary.ssd_read_gib_per_second == 10.0
    assert result.io_summary.planned_read_seconds == pytest.approx(
        400 / (10.0 * 1024**3)
    )
    assert result.io_summary.max_read_seconds == 1.0
    assert result.io_summary.read_seconds_ok is True
    assert result.io_summary.copy_seconds_ok is True
    assert result.io_summary.copy_read_calls == len(pread_calls)
    assert result.io_summary.copy_write_calls == len(pread_calls)
    assert result.io_summary.copy_average_read_bytes == pytest.approx(400 / 8)
    assert result.io_summary.copy_average_write_bytes == pytest.approx(400 / 8)
    assert result.io_summary.copy_read_call_counterfactuals_by_chunk_mib == {
        "8": 2,
        "16": 2,
        "32": 2,
        "64": 2,
        "128": 2,
    }
    assert result.io_summary.max_raw_ranges == 0
    assert result.io_summary.raw_range_count_ok is None
    assert result.io_summary.max_coalesced_ranges == 0
    assert result.io_summary.coalesced_range_count_ok is None
    assert [(item.source_offset, item.stage_offset, item.stage_length) for item in result.ranges] == [
        (0, 0, 300),
        (400, 300, 100),
    ]
    assert [(item.expert, item.stage_offset) for item in result.slots] == [
        (0, 0),
        (1, 100),
        (2, 200),
        (4, 300),
    ]
    staged = stage_file.read_bytes()
    assert staged[0:1] == b"\x00"
    assert staged[100:101] == b"\x01"
    assert staged[200:201] == b"\x02"
    assert staged[300:301] == b"\x04"
    manifest = json.loads(stage_file.with_suffix(".bin.manifest.json").read_text())
    assert manifest["staged_bytes"] == 400
    assert manifest["io_summary"]["planned_read_bytes"] == 400
    assert manifest["io_summary"]["planned_read_seconds"] == pytest.approx(
        400 / (10.0 * 1024**3)
    )
    assert manifest["io_summary"]["read_seconds_ok"] is True
    assert manifest["io_summary"]["raw_range_count"] == 2
    assert manifest["io_summary"]["max_raw_ranges"] == 0
    assert manifest["io_summary"]["raw_range_count_ok"] is None
    assert manifest["io_summary"]["max_coalesced_ranges"] == 0
    assert manifest["io_summary"]["coalesced_range_count_ok"] is None
    assert manifest["io_summary"]["copy_read_calls"] == len(pread_calls)
    assert manifest["io_summary"]["copy_average_read_bytes"] == pytest.approx(400 / 8)
    assert manifest["io_summary"][
        "copy_read_call_counterfactuals_by_chunk_mib"
    ] == {"8": 2, "16": 2, "32": 2, "64": 2, "128": 2}
    assert manifest["slots"][3]["expert"] == 4


def test_stage_batch_experts_uses_expert_order_source_offsets(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        layers[0]["expert_order"] = [4, 1, 0, 2, 3, 5, 6, 7]

    _mutate_layout(layout, mutate)
    (layout.parent / "layer_003.bin").write_bytes(
        b"".join(bytes([expert]) * 100 for expert in (4, 1, 0, 2, 3, 5, 6, 7))
    )
    router_dir = tmp_path / "router_json"
    router_dir.mkdir()
    (router_dir / "token_000000.router.json").write_text(
        json.dumps({"experts": [0, 1, 4], "weights": [0.3, 0.3, 0.4]}),
        encoding="utf-8",
    )
    stage_file = tmp_path / "stage.bin"

    result = stage_batch_experts(
        layout,
        layer=3,
        router_json_dir=router_dir,
        stage_file_path=stage_file,
        merge_gap_bytes=0,
        align_bytes=1,
        max_stage_mib=1,
    )

    assert result.selected_experts == (0, 1, 4)
    assert result.planned_read_bytes == 300
    assert result.io_summary.raw_range_count == 1
    assert sorted(
        (item.expert, item.source_offset, item.stage_offset)
        for item in result.slots
    ) == sorted([
        (4, 0, 0),
        (1, 100, 100),
        (0, 200, 200),
    ])
    staged = stage_file.read_bytes()
    assert staged[0:1] == b"\x04"
    assert staged[100:101] == b"\x01"
    assert staged[200:201] == b"\x00"


def test_stage_batch_experts_rejects_read_seconds_cap_before_output(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"

    with pytest.raises(ExpertIOPlanError, match="planned stage read time"):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=1,
            ssd_read_gib_per_second=1.0,
            max_read_seconds=1e-12,
        )

    assert not stage_file.exists()


def test_stage_batch_experts_rejects_actual_copy_seconds_over_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"
    manifest = tmp_path / "stage.manifest.json"
    ticks = iter((10.0, 11.5))

    monkeypatch.setattr(
        expert_io_module.time,
        "perf_counter",
        lambda: next(ticks),
    )

    with pytest.raises(
        ExpertIOPlanError,
        match="actual stage copy time .* exceeds limit 1",
    ):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            manifest_path=manifest,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=1,
            ssd_read_gib_per_second=10.0,
            max_read_seconds=1.0,
        )

    assert not stage_file.exists()
    assert not manifest.exists()


def test_stage_batch_experts_rejects_raw_range_cap_before_output(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"
    manifest = tmp_path / "stage.manifest.json"

    with pytest.raises(ExpertIOPlanError, match="raw range count 2 exceeds limit 1"):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            manifest_path=manifest,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=1,
            max_raw_ranges=1,
        )

    assert not stage_file.exists()
    assert not manifest.exists()


def test_stage_batch_experts_rejects_coalesced_range_cap_before_output(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"
    manifest = tmp_path / "stage.manifest.json"

    with pytest.raises(
        ExpertIOPlanError,
        match="coalesced range count 2 exceeds limit 1",
    ):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            manifest_path=manifest,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=1,
            max_coalesced_ranges=1,
        )

    assert not stage_file.exists()
    assert not manifest.exists()


def test_stage_batch_experts_requires_speed_for_read_seconds_cap(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"

    with pytest.raises(ExpertIOPlanError, match="ssd_read_gib_per_second"):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            max_read_seconds=1.0,
        )

    assert not stage_file.exists()


def test_stage_batch_experts_rejects_stage_limit(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)

    with pytest.raises(ExpertIOPlanError, match="exceed limit"):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=tmp_path / "stage.bin",
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=0.0001,
        )


def test_stage_batch_experts_rejects_non_finite_stage_cap_before_output(
    tmp_path: Path,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"

    with pytest.raises(ExpertIOPlanError, match="max_stage_mib"):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            max_stage_mib=float("nan"),
        )

    assert not stage_file.exists()


def test_stage_batch_experts_rejects_stage_disk_budget(tmp_path: Path) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"

    with pytest.raises(ExpertIOPlanError, match="not enough free disk"):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=1,
            disk_safety_margin_bytes=10**30,
        )

    assert not stage_file.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"disk_safety_margin_bytes": True},
            "disk_safety_margin_bytes must be an integer",
        ),
        (
            {"disk_safety_margin_bytes": 1.5},
            "disk_safety_margin_bytes must be an integer",
        ),
        (
            {"disk_safety_margin_bytes": -1},
            "disk_safety_margin_bytes must be non-negative",
        ),
        (
            {"max_raw_ranges": True},
            "max_raw_ranges must be an integer",
        ),
        (
            {"max_raw_ranges": -1},
            "max_raw_ranges must be non-negative",
        ),
        (
            {"max_coalesced_ranges": False},
            "max_coalesced_ranges must be an integer",
        ),
        (
            {"max_coalesced_ranges": -1},
            "max_coalesced_ranges must be non-negative",
        ),
    ),
)
def test_stage_batch_experts_rejects_invalid_integer_stage_guards(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"

    with pytest.raises(ExpertIOPlanError, match=message):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=1,
            **kwargs,
        )

    assert not stage_file.exists()


def test_stage_batch_experts_removes_partial_stage_file_on_copy_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"

    def fail_copy(**kwargs) -> None:
        kwargs["destination"].write(b"partial")
        raise ExpertIOPlanError("copy exploded")

    monkeypatch.setattr("largerlm.expert_io._copy_exact_range", fail_copy)

    with pytest.raises(ExpertIOPlanError, match="failed to stage batch experts"):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=1,
        )

    assert not stage_file.exists()


def test_write_json_atomic_removes_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "stage.manifest.json"

    def fail_replace(self: Path, target: Path) -> None:
        raise OSError("replace exploded")

    monkeypatch.setattr(type(manifest), "replace", fail_replace)

    with pytest.raises(OSError, match="replace exploded"):
        expert_io_module._write_json_atomic(manifest, {"ok": True})

    assert not manifest.exists()
    assert not manifest.with_name(manifest.name + ".tmp").exists()


def test_stage_batch_experts_removes_stage_file_on_manifest_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"
    manifest = tmp_path / "stage_manifest.json"

    def fail_manifest(path: Path, payload: object) -> None:
        raise OSError("manifest exploded")

    monkeypatch.setattr(expert_io_module, "_write_json_atomic", fail_manifest)

    with pytest.raises(ExpertIOPlanError, match="failed to write stage manifest"):
        stage_batch_experts(
            layout,
            layer=3,
            router_json_dir=router_dir,
            stage_file_path=stage_file,
            manifest_path=manifest,
            merge_gap_bytes=0,
            align_bytes=1,
            max_stage_mib=1,
        )

    assert not stage_file.exists()
    assert not manifest.exists()


def test_stage_batch_experts_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_expert_layout(tmp_path)
    router_dir = _write_router_jsons(tmp_path)
    stage_file = tmp_path / "stage.bin"
    manifest = tmp_path / "stage_manifest.json"

    status = cli_main(
        [
            "stage-batch-experts",
            str(layout),
            "--layer",
            "3",
            "--router-json-dir",
            str(router_dir),
            "--stage-file",
            str(stage_file),
            "--manifest",
            str(manifest),
            "--align-kib",
            "0.0009765625",
            "--max-stage-mib",
            "1",
            "--copy-chunk-mib",
            "0.00005",
            "--stage-disk-margin-mib",
            "0",
            "--ssd-read-gib-s",
            "10",
            "--max-read-seconds",
            "1",
            "--max-raw-ranges",
            "2",
            "--max-coalesced-ranges",
            "2",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["read_advice"]["attempted_ranges"] == 2
    assert payload["read_advice"]["advised_bytes"] in {0, 400}
    assert payload["staged_bytes"] == 400
    assert payload["io_summary"]["serial_read_bytes"] == 500
    assert payload["io_summary"]["coalesced_savings_bytes"] == 100
    assert payload["io_summary"]["planned_read_seconds"] == pytest.approx(
        400 / (10.0 * 1024**3)
    )
    assert payload["io_summary"]["max_read_seconds"] == 1.0
    assert payload["io_summary"]["read_seconds_ok"] is True
    assert payload["io_summary"]["copy_seconds_ok"] is True
    assert payload["io_summary"]["max_raw_ranges"] == 2
    assert payload["io_summary"]["raw_range_count_ok"] is True
    assert payload["io_summary"]["max_coalesced_ranges"] == 2
    assert payload["io_summary"]["coalesced_range_count_ok"] is True
    assert payload["stage_file_path"] == str(stage_file)
    assert payload["manifest_path"] == str(manifest)
    assert manifest.exists()
