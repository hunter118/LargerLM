from __future__ import annotations

import json
from pathlib import Path

import pytest

from largerlm.runtime_check import RuntimeCheckError, check_layer_runtime


def _write_layouts(
    root: Path,
    *,
    router_hidden: int = 8,
    include_shared: bool = False,
    include_attention: bool = False,
    decoder_attention: bool = False,
    include_dense: bool = False,
) -> tuple[Path, Path]:
    expert_dir = root / "experts"
    resident_dir = root / "resident"
    expert_dir.mkdir()
    resident_dir.mkdir()

    components = [
        ("gate_proj.weight", 0, 32, "U32", [8, 1]),
        ("gate_proj.scales", 32, 16, "BF16", [8, 1]),
        ("gate_proj.biases", 48, 16, "BF16", [8, 1]),
        ("up_proj.weight", 64, 32, "U32", [8, 1]),
        ("up_proj.scales", 96, 16, "BF16", [8, 1]),
        ("up_proj.biases", 112, 16, "BF16", [8, 1]),
        ("down_proj.weight", 128, 32, "U32", [8, 1]),
        ("down_proj.scales", 160, 16, "BF16", [8, 1]),
        ("down_proj.biases", 176, 16, "BF16", [8, 1]),
    ]
    expert_layout = expert_dir / "layout.json"
    expert_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "quantization": "mlx-affine-int4",
                "group_size": 8,
                "num_layers": 2,
                "num_experts": 2,
                "component_order": [name for name, *_ in components],
                "layers": [
                    {
                        "layer": 1,
                        "num_experts": 2,
                        "expert_slot_bytes": 192,
                        "layer_file": "layer_001.bin",
                        "components": [
                            {
                                "name": name,
                                "offset": offset,
                                "size": size,
                                "dtype": dtype,
                                "shape": shape,
                            }
                            for name, offset, size, dtype, shape in components
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    resident_layout = resident_dir / "layout.json"
    tensors = [
        {
            "name": "model.layers.1.mlp.gate.weight",
            "offset": 0,
            "size": 64,
            "dtype": "F32",
            "shape": [2, router_hidden],
            "category": "routers",
        }
    ]
    offset = 64
    if include_dense:
        for component in ("gate_proj", "up_proj", "down_proj"):
            tensors.append(
                {
                    "name": f"model.layers.0.mlp.switch_mlp.{component}.weight",
                    "offset": offset,
                    "size": 256,
                    "dtype": "F32",
                    "shape": [8, 8],
                    "category": "dense_mlp",
                }
            )
            offset += 256
    if include_shared:
        for component in ("gate_proj", "up_proj", "down_proj"):
            tensors.append(
                {
                    "name": f"model.layers.1.mlp.shared_experts.{component}.weight",
                    "offset": offset,
                    "size": 256,
                    "dtype": "F32",
                    "shape": [8, 8],
                    "category": "shared_experts",
                }
            )
            offset += 256
    if include_attention:
        if decoder_attention:
            attention_tensors = [
                ("model.layers.1.input_layernorm.weight", 32, "F32", [8], "norms"),
                ("model.layers.1.self_attn.q_a_layernorm.weight", 8, "F32", [2], "norms"),
                ("model.layers.1.self_attn.kv_a_layernorm.weight", 8, "F32", [2], "norms"),
                ("model.layers.1.post_attention_layernorm.weight", 32, "F32", [8], "norms"),
                ("model.layers.1.self_attn.q_a_proj.weight", 64, "F32", [2, 8], "attention"),
                ("model.layers.1.self_attn.q_b_proj.weight", 48, "F32", [6, 2], "attention"),
                (
                    "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                    128,
                    "F32",
                    [4, 8],
                    "attention",
                ),
                ("model.layers.1.self_attn.kv_b_proj.weight", 32, "F32", [4, 2], "attention"),
                ("model.layers.1.self_attn.o_proj.weight", 64, "F32", [8, 2], "attention"),
            ]
        else:
            attention_tensors = [
                ("model.layers.1.input_layernorm.weight", 32, "F32", [8], "norms"),
                ("model.layers.1.self_attn.q_a_layernorm.weight", 8, "F32", [2], "norms"),
                ("model.layers.1.self_attn.kv_a_layernorm.weight", 8, "F32", [2], "norms"),
                ("model.layers.1.post_attention_layernorm.weight", 32, "F32", [8], "norms"),
                ("model.layers.1.self_attn.q_a_proj.weight", 64, "F32", [2, 8], "attention"),
                ("model.layers.1.self_attn.q_b_proj.weight", 16, "BF16", [4, 2], "attention"),
                (
                    "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
                    96,
                    "F32",
                    [3, 8],
                    "attention",
                ),
                ("model.layers.1.self_attn.kv_b_proj.weight", 32, "F32", [4, 2], "attention"),
            ]
        for name, size, dtype, shape, category in attention_tensors:
            tensors.append(
                {
                    "name": name,
                    "offset": offset,
                    "size": size,
                    "dtype": dtype,
                    "shape": shape,
                    "category": category,
                }
            )
            offset += size
    resident_layout.write_text(
        json.dumps(
            {
                "version": 1,
                "model_type": "glm_moe_dsa",
                "alignment": 64,
                "weight_file": "resident.bin",
                "total_bytes": offset,
                "tensors": tensors,
            }
        ),
        encoding="utf-8",
    )
    return expert_layout, resident_layout


def _mutate_layout(path: Path, mutator: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_layer_runtime_estimates_safe_peak(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
    )

    assert budget.hidden_dim == 8
    assert budget.intermediate_dim == 8
    assert budget.expert_slot_bytes == 192
    assert budget.aligned_slot_bytes == 2 * 1024 * 1024
    assert budget.read_bytes_per_token == 384
    assert budget.estimated_peak_bytes == budget.moe_stage_peak_bytes


def test_check_layer_runtime_ignores_router_bias_when_finding_gate(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors.insert(
            0,
            {
                "name": "model.layers.1.mlp.gate.e_score_correction_bias",
                "offset": 64,
                "size": 8,
                "dtype": "F32",
                "shape": [2],
                "category": "routers",
            },
        )
        payload["total_bytes"] = 72

    _mutate_layout(resident_layout, mutate)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
    )

    assert budget.router_bytes == 64


def test_check_layer_runtime_accepts_affine_int4_router(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0].update({"size": 8, "dtype": "U32", "shape": [2, 1]})
        tensors.append(
            {
                "name": "model.layers.1.mlp.gate.scales",
                "offset": 8,
                "size": 4,
                "dtype": "BF16",
                "shape": [2, 1],
                "category": "routers",
            }
        )
        tensors.append(
            {
                "name": "model.layers.1.mlp.gate.biases",
                "offset": 12,
                "size": 4,
                "dtype": "BF16",
                "shape": [2, 1],
                "category": "routers",
            }
        )

    _mutate_layout(resident_layout, mutate)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=16,
    )

    assert budget.router_bytes == 16
    assert budget.router_stage_peak_bytes == (
        2 * 2 * 1024 * 1024 + 2 * 8 * 4 + 2 * 2 * 4
    )

    with pytest.raises(RuntimeCheckError, match="router tensor 16 bytes exceeds limit 15"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
            max_slot_bytes=1024,
            max_router_bytes=15,
        )


def test_check_layer_runtime_accepts_mxfp4_router(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0].update({"size": 8, "dtype": "U32", "shape": [2, 1]})
        tensors.append(
            {
                "name": "model.layers.1.mlp.gate.scales",
                "offset": 8,
                "size": 2,
                "dtype": "U8",
                "shape": [2, 1],
                "category": "routers",
            }
        )

    _mutate_layout(resident_layout, mutate)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=10,
    )

    assert budget.router_bytes == 10
    assert budget.router_stage_peak_bytes == (
        2 * 2 * 1024 * 1024 + 2 * 8 * 4 + 2 * 2 * 4
    )

    with pytest.raises(RuntimeCheckError, match="router tensor 10 bytes exceeds limit 9"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
            max_slot_bytes=1024,
            max_router_bytes=9,
        )


def test_check_layer_runtime_estimates_dense_mlp_peak(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path, include_dense=True)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=0,
        dense_mlp=True,
        max_resident_matrix_bytes=1024,
        max_runner_scratch_bytes=64 * 1024**2,
    )

    assert budget.layer_kind == "dense"
    assert budget.hidden_dim == 8
    assert budget.intermediate_dim == 8
    assert budget.num_experts == 0
    assert budget.top_k == 0
    assert budget.expert_slot_bytes == 0
    assert budget.read_bytes_per_token == 0
    assert budget.moe_stage_peak_bytes >= 2 * 1024 * 1024
    assert budget.estimated_peak_bytes == budget.moe_stage_peak_bytes


def test_check_layer_runtime_rejects_topk_above_cap(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    with pytest.raises(RuntimeCheckError, match="top_k"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=3,
            max_k=2,
        )


def test_check_layer_runtime_rejects_router_shape_mismatch(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path, router_hidden=7)

    with pytest.raises(RuntimeCheckError, match="router hidden dim"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
        )


def test_check_layer_runtime_rejects_boolean_expert_shape(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        components = layers[0]["components"]
        components[0]["shape"][1] = True

    _mutate_layout(expert_layout, mutate)

    with pytest.raises(RuntimeCheckError, match="2-D integer shape"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
        )


def test_check_layer_runtime_rejects_boolean_expert_layer_field(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        layers[0]["expert_slot_bytes"] = True

    _mutate_layout(expert_layout, mutate)

    with pytest.raises(
        RuntimeCheckError,
        match="expert layer expert_slot_bytes must be an integer",
    ):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
        )


def test_check_layer_runtime_rejects_boolean_expert_layer_id(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        layers = payload["layers"]
        assert isinstance(layers, list)
        layers[0]["layer"] = True

    _mutate_layout(expert_layout, mutate)

    with pytest.raises(RuntimeCheckError, match="layer 1 not found"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
        )


def test_check_layer_runtime_rejects_boolean_router_shape(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["shape"][1] = False

    _mutate_layout(resident_layout, mutate)

    with pytest.raises(RuntimeCheckError, match="router tensor must have shape"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
        )


def test_check_layer_runtime_rejects_boolean_router_size(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[0]["size"] = True

    _mutate_layout(resident_layout, mutate)

    with pytest.raises(
        RuntimeCheckError,
        match="router tensor size must be an integer",
    ):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
        )


def test_check_layer_runtime_includes_shared_expert_peak(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path, include_shared=True)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
        include_shared_expert=True,
    )

    assert budget.include_shared_expert is True
    assert budget.shared_intermediate_dim == 8
    assert budget.shared_max_matrix_bytes == 256
    assert budget.shared_max_aligned_matrix_bytes == 2 * 1024 * 1024
    assert budget.moe_stage_peak_bytes > 2 * 1024 * 1024


def test_check_layer_runtime_accepts_affine_int4_shared_expert(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path, include_shared=True)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        next_offset = 1024
        rewritten: list[dict[str, object]] = []
        for tensor in tensors:
            name = tensor.get("name")
            if isinstance(name, str) and ".shared_experts." in name:
                tensor = dict(tensor)
                tensor.update({"offset": next_offset, "size": 32, "dtype": "U32", "shape": [8, 1]})
                rewritten.append(tensor)
                next_offset += 32
                stem = name[: -len(".weight")]
                rewritten.append(
                    {
                        "name": f"{stem}.scales",
                        "offset": next_offset,
                        "size": 16,
                        "dtype": "BF16",
                        "shape": [8, 1],
                        "category": "shared_experts",
                    }
                )
                next_offset += 16
                rewritten.append(
                    {
                        "name": f"{stem}.biases",
                        "offset": next_offset,
                        "size": 16,
                        "dtype": "BF16",
                        "shape": [8, 1],
                        "category": "shared_experts",
                    }
                )
                next_offset += 16
            else:
                rewritten.append(tensor)
        payload["tensors"] = rewritten

    _mutate_layout(resident_layout, mutate)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
        include_shared_expert=True,
    )

    assert budget.include_shared_expert is True
    assert budget.shared_intermediate_dim == 8
    assert budget.shared_max_matrix_bytes == 64
    assert budget.shared_max_aligned_matrix_bytes == 2 * 1024 * 1024


def test_check_layer_runtime_includes_attention_projection_peak(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path, include_attention=True)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
        max_resident_matrix_bytes=1024,
        include_attention_projections=True,
    )

    assert budget.include_attention_projections is True
    assert budget.attention_q_lora_dim == 2
    assert budget.attention_q_output_dim == 4
    assert budget.attention_kv_lora_dim == 2
    assert budget.attention_kv_rope_dim == 1
    assert budget.attention_kv_output_dim == 4
    assert budget.attention_read_bytes_per_token == 64 + 16 + 96 + 32
    assert budget.attention_max_matrix_bytes == 96
    assert budget.attention_max_aligned_matrix_bytes == 2 * 1024 * 1024
    assert budget.attention_projection_peak_bytes > 2 * 1024 * 1024


def test_check_layer_runtime_rejects_attention_matrix_over_cap(tmp_path: Path) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path, include_attention=True)

    with pytest.raises(RuntimeCheckError, match="resident attention matrix"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
            max_slot_bytes=1024,
            max_router_bytes=1024,
            max_resident_matrix_bytes=64,
            include_attention_projections=True,
        )


def test_check_layer_runtime_rejects_boolean_resident_vector_shape(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path, include_attention=True)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        for tensor in tensors:
            if tensor["name"].endswith(".input_layernorm.weight"):
                tensor["shape"][0] = True
                return
        raise AssertionError("input layernorm fixture tensor not found")

    _mutate_layout(resident_layout, mutate)

    with pytest.raises(
        RuntimeCheckError,
        match="input_layernorm must have a 1-D integer shape",
    ):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
            include_attention_projections=True,
        )


def test_check_layer_runtime_rejects_boolean_resident_matrix_shape(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(tmp_path, include_attention=True)

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        for tensor in tensors:
            if tensor["name"].endswith(".self_attn.q_a_proj.weight"):
                tensor["shape"][0] = False
                return
        raise AssertionError("q_a projection fixture tensor not found")

    _mutate_layout(resident_layout, mutate)

    with pytest.raises(
        RuntimeCheckError,
        match="q_a_proj.weight must have a 2-D integer shape",
    ):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
            include_attention_projections=True,
        )


def test_check_layer_runtime_includes_decoder_layer_cache_and_attention_peaks(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(
        tmp_path,
        include_shared=True,
        include_attention=True,
        decoder_attention=True,
    )

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
        max_resident_matrix_bytes=1024,
        max_cache_read_bytes=1024,
        include_shared_expert=True,
        include_decoder_layer=True,
        context_length=16,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
    )

    assert budget.include_attention_projections is True
    assert budget.include_decoder_layer is True
    assert budget.decoder_cache_read_bytes == 16 * (2 + 2) * 2
    assert budget.decoder_cache_f32_bytes == 16 * (2 + 2) * 4
    assert budget.dsa_indexer_mode == "none"
    assert budget.dsa_index_cache_read_bytes == 0
    assert budget.decoder_mla_cache_read_bytes == budget.decoder_cache_read_bytes
    assert budget.decoder_mla_attention_peak_bytes > budget.decoder_cache_read_bytes
    assert budget.decoder_attention_output_peak_bytes >= 2 * 1024 * 1024
    assert budget.estimated_peak_bytes >= budget.decoder_attention_output_peak_bytes


def test_check_layer_runtime_accepts_absorbed_attention_aliases(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(
        tmp_path,
        include_shared=True,
        include_attention=True,
        decoder_attention=True,
    )

    def mutate(payload: dict[str, object]) -> None:
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        tensors[:] = [
            tensor
            for tensor in tensors
            if not tensor["name"].endswith(".self_attn.kv_b_proj.weight")
        ]
        offset = int(payload["total_bytes"])
        for name, shape in (
            ("model.layers.1.self_attn.embed_q.weight", [2, 2, 1]),
            ("model.layers.1.self_attn.unembed_out.weight", [2, 1, 2]),
        ):
            size = 4
            for dim in shape:
                size *= dim
            tensors.append(
                {
                    "name": name,
                    "offset": offset,
                    "size": size,
                    "dtype": "F32",
                    "shape": shape,
                    "category": "attention",
                }
            )
            offset += size
        payload["total_bytes"] = offset

    _mutate_layout(resident_layout, mutate)

    budget = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
        max_resident_matrix_bytes=1024,
        max_cache_read_bytes=1024,
        include_shared_expert=True,
        include_decoder_layer=True,
        context_length=16,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
    )

    assert budget.attention_kv_lora_dim == 2
    assert budget.attention_kv_output_dim == 4
    assert budget.attention_read_bytes_per_token == 64 + 48 + 128
    assert budget.decoder_mla_attention_peak_bytes == 640


def test_check_layer_runtime_uses_dsa_indexed_cache_budget(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(
        tmp_path,
        include_attention=True,
        decoder_attention=True,
    )

    full = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
        max_resident_matrix_bytes=1024,
        max_cache_read_bytes=1024,
        include_decoder_layer=True,
        context_length=16,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        dsa_indexer_mode="full",
        dsa_index_topk=4,
        dsa_index_head_dim=2,
    )
    shared = check_layer_runtime(
        expert_layout,
        resident_layout,
        layer=1,
        top_k=2,
        max_k=2,
        max_slot_bytes=1024,
        max_router_bytes=1024,
        max_resident_matrix_bytes=1024,
        max_cache_read_bytes=1024,
        include_decoder_layer=True,
        context_length=16,
        num_heads=2,
        qk_nope_dim=1,
        rope_dim=2,
        v_head_dim=1,
        dsa_indexer_mode="shared",
        dsa_index_topk=4,
    )

    assert full.dsa_index_cache_read_bytes == 16 * 2 * 2
    assert full.decoder_mla_cache_read_bytes == 4 * (2 + 2) * 2
    assert full.decoder_cache_read_bytes == 96
    assert full.decoder_cache_f32_bytes == 4 * (2 + 2) * 4
    assert shared.dsa_index_cache_read_bytes == 0
    assert shared.decoder_mla_cache_read_bytes == 4 * (2 + 2) * 2
    assert shared.decoder_cache_read_bytes == 32
    assert shared.decoder_cache_f32_bytes == 4 * (2 + 2) * 4


def test_check_layer_runtime_requires_dsa_index_head_dim_for_full(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(
        tmp_path,
        include_attention=True,
        decoder_attention=True,
    )

    with pytest.raises(RuntimeCheckError, match="dsa_index_head_dim"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
            max_slot_bytes=1024,
            max_router_bytes=1024,
            max_resident_matrix_bytes=1024,
            max_cache_read_bytes=1024,
            include_decoder_layer=True,
            context_length=16,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
            dsa_indexer_mode="full",
            dsa_index_topk=4,
        )


def test_check_layer_runtime_rejects_decoder_cache_read_over_cap(
    tmp_path: Path,
) -> None:
    expert_layout, resident_layout = _write_layouts(
        tmp_path,
        include_attention=True,
        decoder_attention=True,
    )

    with pytest.raises(RuntimeCheckError, match="decoder cache read"):
        check_layer_runtime(
            expert_layout,
            resident_layout,
            layer=1,
            top_k=2,
            max_k=2,
            max_slot_bytes=1024,
            max_router_bytes=1024,
            max_resident_matrix_bytes=1024,
            max_cache_read_bytes=64,
            include_decoder_layer=True,
            context_length=16,
            num_heads=2,
            qk_nope_dim=1,
            rope_dim=2,
            v_head_dim=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layer": True}, "layer must be an integer"),
        ({"top_k": False}, "top_k must be an integer"),
        ({"max_k": True}, "max_k must be an integer"),
        ({"max_slot_bytes": True}, "max_slot_bytes must be an integer"),
        ({"context_length": True}, "context_length must be an integer"),
        ({"cache_dtype_bytes": False}, "cache_dtype_bytes must be an integer"),
        ({"dsa_index_topk": True}, "dsa_index_topk must be an integer"),
    ],
)
def test_check_layer_runtime_rejects_boolean_integer_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    expert_layout, resident_layout = _write_layouts(
        tmp_path,
        include_attention=True,
        decoder_attention=True,
    )

    args: dict[str, object] = {
        "layer": 1,
        "top_k": 2,
        "max_k": 2,
        "max_slot_bytes": 1024,
        "max_router_bytes": 1024,
        "max_resident_matrix_bytes": 1024,
        "max_cache_read_bytes": 1024,
        "include_decoder_layer": True,
        "context_length": 16,
        "num_heads": 2,
        "qk_nope_dim": 1,
        "rope_dim": 2,
        "v_head_dim": 1,
        "cache_dtype_bytes": 2,
        "dsa_indexer_mode": "shared",
        "dsa_index_topk": 4,
    }
    args.update(kwargs)

    with pytest.raises(RuntimeCheckError, match=message):
        check_layer_runtime(expert_layout, resident_layout, **args)
