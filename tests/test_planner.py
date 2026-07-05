from __future__ import annotations

import json
from pathlib import Path

import pytest

from largerlm.config import ConfigError
from largerlm.config import load_config
from largerlm.cli import main as cli_main
from largerlm.planner import (
    ExpertLayout,
    PlannerError,
    build_plan,
    default_system_reserve_bytes,
    estimate_config_resident_bytes,
    estimate_mla_cache,
)
from largerlm.safetensors import CheckpointStats
from largerlm.safetensors import categorize_tensor, is_routed_expert_tensor
from largerlm.server import _is_public_glm_5_2_shape


FIXTURES = Path(__file__).parent / "fixtures"


def test_expert_layout_matches_affine_quant_math() -> None:
    layout = ExpertLayout(hidden_size=128, intermediate_size=256, weight_bits=4)

    # gate/up: 256x128 4-bit weights plus 256 rows * 2 groups * (scale+bias bf16)
    assert layout.gate.weight_bytes == 16_384
    assert layout.gate.metadata_bytes == 2_048
    assert layout.up.total_bytes == layout.gate.total_bytes

    # down: 128x256 4-bit weights plus 128 rows * 4 groups * (scale+bias bf16)
    assert layout.down.weight_bytes == 16_384
    assert layout.down.metadata_bytes == 2_048
    assert layout.total_bytes == 55_296


def test_glm_config_plan_counts_moe_layers_and_io() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    plan = build_plan(cfg, quant_bits=4, group_size=64, unified_memory_bytes=128 * 1024**3)

    assert cfg.num_moe_layers == 5
    assert plan.routed_expert_disk_bytes_estimate == 5 * 4 * 55_296
    assert plan.routed_expert_read_bytes_per_decode_token == 5 * 2 * 55_296
    assert plan.suggested_decode_guard_flags is not None
    assert plan.suggested_decode_guard_flags["source"] == "plan"
    assert plan.suggested_decode_guard_flags["decode_read_bytes_per_token"] == (
        5 * 2 * 55_296
    )
    assert "--decode-max-routed-read-gib-per-token" in plan.suggested_decode_guard_flags[
        "argv"
    ]
    assert plan.suggested_launch_guard_flags is not None
    assert plan.suggested_launch_guard_flags["source"] == "plan"
    assert plan.suggested_launch_guard_flags["require_prepared_memory_profile"] is True
    assert plan.suggested_launch_guard_flags[
        "recommended_max_live_working_set_bytes"
    ] == 8 * 1024**3
    assert plan.suggested_launch_guard_flags[
        "recommended_min_free_unified_memory_bytes"
    ] == 24 * 1024**3
    assert plan.suggested_launch_guard_flags[
        "recommended_required_available_memory_bytes"
    ] == 32 * 1024**3
    assert plan.suggested_launch_guard_flags["argv"] == (
        "--require-prepared-memory-profile",
        "--max-live-working-set-mib",
        "8192",
        "--min-free-unified-memory-gib",
        "24",
    )
    assert plan.suggested_launch_profile is not None
    assert plan.suggested_launch_profile["source"] == "plan"
    assert plan.suggested_launch_profile["argv_safe_to_replay"] is True
    assert plan.suggested_launch_profile["sections"]["launch_guard_flags"] == (
        plan.suggested_launch_guard_flags
    )
    assert plan.suggested_launch_profile["sections"]["decode_guard_flags"] == (
        plan.suggested_decode_guard_flags
    )
    assert plan.suggested_launch_profile["argv"] == (
        plan.suggested_launch_guard_flags["argv"]
        + plan.suggested_decode_guard_flags["argv"]
    )
    assert plan.suggested_prepare_flags is not None
    assert plan.suggested_prepare_flags["source"] == "plan"
    assert plan.suggested_prepare_flags["argv_safe_to_replay"] is True
    assert plan.suggested_prepare_flags["context_selection"] == (
        "auto_context_from_budget"
    )
    assert plan.suggested_prepare_flags["group_size"] == 64
    assert plan.suggested_prepare_flags["decode_cache_budget_bytes"] == (
        plan.decode_cache_budget_bytes
    )
    assert plan.suggested_prepare_flags["decode_cache_safe_context_tokens"] == (
        plan.decode_cache_safe_context_tokens
    )
    assert "--auto-context-from-budget" in plan.suggested_prepare_flags["argv"]
    assert "--max-cache-gib" in plan.suggested_prepare_flags["argv"]
    assert "--unified-memory-gib" in plan.suggested_prepare_flags["argv"]
    assert "--page-cache-fraction" in plan.suggested_prepare_flags["argv"]
    prepare_argv = tuple(plan.suggested_prepare_flags["argv"])
    max_cache_gib = float(
        prepare_argv[prepare_argv.index("--max-cache-gib") + 1]
    )
    assert int(max_cache_gib * 1024**3) >= plan.decode_cache_budget_bytes
    assert plan.resident_bytes_estimate_source == "config_estimate"
    assert plan.page_cache_budget_bytes is not None


def test_checkpoint_scan_resident_bytes_override_config_estimate() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    stats = CheckpointStats(
        total_bytes=12_345,
        routed_expert_bytes=2_000,
        resident_bytes=10_000,
        unknown_bytes=0,
        tensor_count=3,
        by_category={"resident_other": 10_000},
    )
    plan = build_plan(
        cfg,
        quant_bits=4,
        group_size=64,
        checkpoint_stats=stats,
        unified_memory_bytes=128 * 1024**3,
    )

    assert plan.resident_bytes_estimate == 10_000
    assert plan.resident_bytes_estimate_source == "checkpoint_scan"


def test_plan_reports_resident_memory_budget_pressure() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    stats = CheckpointStats(
        total_bytes=90 * 1024**3,
        routed_expert_bytes=1,
        resident_bytes=80 * 1024**3,
        unknown_bytes=0,
        tensor_count=3,
        by_category={"resident_other": 80 * 1024**3},
    )

    plan = build_plan(
        cfg,
        checkpoint_stats=stats,
        unified_memory_bytes=64 * 1024**3,
        runtime_buffer_bytes=8 * 1024**3,
    )

    assert plan.unified_memory_bytes == 64 * 1024**3
    assert plan.system_reserve_bytes == 16 * 1024**3
    assert plan.runtime_buffer_bytes == 8 * 1024**3
    assert plan.resident_memory_budget_bytes == 40 * 1024**3
    assert plan.resident_memory_pressure_bytes == 104 * 1024**3
    assert plan.resident_memory_headroom_bytes == -40 * 1024**3
    assert plan.resident_memory_fits_budget is False
    assert plan.page_cache_budget_bytes == 0


def test_default_system_reserve_scales_for_large_unified_memory() -> None:
    assert default_system_reserve_bytes(None) == 16 * 1024**3
    assert default_system_reserve_bytes(64 * 1024**3) == 16 * 1024**3
    assert default_system_reserve_bytes(128 * 1024**3) == 24 * 1024**3


def test_glm_config_plan_checks_decode_cache_budget() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    plan = build_plan(
        cfg,
        quant_bits=4,
        group_size=64,
        unified_memory_bytes=128 * 1024**3,
        max_context_tokens=1024,
        max_cache_bytes=1024**3,
    )

    assert plan.decode_cache_bytes_estimate == (6 * (16 + 16) + 3 * 16) * 2 * 1024
    assert plan.decode_cache_budget_bytes == 1024**3
    assert plan.decode_cache_fits_budget is True

    tiny_budget = build_plan(
        cfg,
        quant_bits=4,
        group_size=64,
        max_context_tokens=1024,
        max_cache_bytes=1,
    )
    assert tiny_budget.decode_cache_fits_budget is False
    assert tiny_budget.suggested_prepare_flags is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"quant_bits": 0}, "quant_bits"),
        ({"group_size": 0}, "group_size"),
        ({"unified_memory_bytes": 0}, "unified_memory_bytes"),
        ({"system_reserve_bytes": -1}, "system_reserve_bytes"),
        ({"runtime_buffer_bytes": 0}, "runtime_buffer_bytes"),
        ({"target_page_cache_fraction": -0.1}, "target_page_cache_fraction"),
        ({"target_page_cache_fraction": 1.1}, "target_page_cache_fraction"),
        ({"max_context_tokens": 0}, "max_context_tokens"),
        ({"max_cache_bytes": -1}, "max_cache_bytes"),
        ({"cold_read_gib_per_second": 0.0}, "cold_read_gib_per_second"),
    ),
)
def test_build_plan_rejects_invalid_budget_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")

    with pytest.raises(PlannerError, match=message):
        build_plan(cfg, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"quant_bits": True}, "quant_bits must be an integer"),
        ({"quant_bits": 4.5}, "quant_bits must be an integer"),
        ({"group_size": False}, "group_size must be an integer"),
        ({"group_size": 64.5}, "group_size must be an integer"),
        ({"unified_memory_bytes": True}, "unified_memory_bytes must be an integer"),
        ({"system_reserve_bytes": 1.5}, "system_reserve_bytes must be an integer"),
        ({"runtime_buffer_bytes": False}, "runtime_buffer_bytes must be an integer"),
        ({"max_context_tokens": 1.5}, "max_context_tokens must be an integer"),
        ({"max_cache_bytes": True}, "max_cache_bytes must be an integer"),
    ),
)
def test_build_plan_rejects_non_integer_budget_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")

    with pytest.raises(PlannerError, match=message):
        build_plan(cfg, **kwargs)


def test_mla_cache_estimate_handles_dsa_indexer_layers() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")
    cache = estimate_mla_cache(cfg)

    assert cache.indexer_full_layers == 3
    assert cache.bytes_per_token == (6 * (16 + 16) + 3 * 16) * 2
    assert cache.mla_cache_width == 32
    assert cache.indexer_bytes_per_token == 3 * 16 * 2


def test_glm_mla_attention_dims_are_derived_from_config() -> None:
    cfg = load_config(FIXTURES / "glm_moe_dsa_config.json")

    assert cfg.num_attention_heads == 4
    assert cfg.q_lora_rank == 32
    assert cfg.kv_lora_rank == 16
    assert cfg.qk_nope_head_dim == 16
    assert cfg.qk_rope_head_dim == 16
    assert cfg.v_head_dim == 16
    assert cfg.attention_q_head_dim == 32
    assert cfg.attention_q_projection_output_dim == 128
    assert cfg.attention_kv_a_output_dim == 32
    assert cfg.attention_kv_b_output_dim == 128
    assert cfg.attention_value_output_dim == 64
    assert cfg.mla_cache_width == 32
    assert cfg.index_head_dim == 16
    assert cfg.index_n_heads == 4
    assert cfg.index_topk == 4
    assert cfg.indexer_rope_interleave is False
    assert cfg.dsa_full_indexer_q_output_dim == 64
    assert cfg.max_position_embeddings == 1024
    assert cfg.rms_norm_eps == 0.00001
    assert cfg.rope_theta == 10000.0


def test_glm_5_2_config_uses_explicit_mlp_layer_types() -> None:
    cfg = load_config(FIXTURES / "glm_5_2_config.json")
    plan = build_plan(
        cfg,
        quant_bits=4,
        group_size=64,
        unified_memory_bytes=128 * 1024**3,
        max_context_tokens=cfg.max_position_embeddings,
    )

    assert cfg.num_hidden_layers == 78
    assert _is_public_glm_5_2_shape(cfg) is True
    assert cfg.vocab_size == 154880
    assert cfg.eos_token_ids == (154820, 154827, 154829)
    assert cfg.weight_dtype == "bfloat16"
    assert cfg.weight_dtype_bytes == 2
    assert cfg.num_attention_heads == 64
    assert cfg.num_key_value_heads == 64
    assert cfg.q_lora_rank == 2048
    assert cfg.kv_lora_rank == 512
    assert cfg.qk_nope_head_dim == 192
    assert cfg.qk_rope_head_dim == 64
    assert cfg.v_head_dim == 256
    assert cfg.n_routed_experts == 256
    assert cfg.n_shared_experts == 1
    assert cfg.num_experts_per_tok == 8
    assert cfg.tie_word_embeddings is False
    assert cfg.mlp_layer_types is not None
    assert cfg.moe_layers == list(range(3, 78))
    assert cfg.num_moe_layers == 75
    assert cfg.indexer_types is not None
    assert sum(1 for item in cfg.indexer_types if item == "full") == 21
    assert cfg.index_head_dim == 128
    assert cfg.index_n_heads == 32
    assert cfg.index_topk == 2048
    assert cfg.indexer_rope_interleave is True
    assert cfg.rope_interleave is True
    assert cfg.scoring_func == "sigmoid"
    assert cfg.topk_method == "noaux_tc"
    assert cfg.norm_topk_prob is True
    assert cfg.routed_scaling_factor == 2.5
    assert cfg.n_group == 1
    assert cfg.topk_group == 1
    assert cfg.dsa_full_indexer_q_output_dim == 4096
    assert cfg.rope_theta == 8000000.0
    assert cfg.max_position_embeddings == 1_048_576
    assert plan.expert_layout.total_bytes == 21_233_664
    assert plan.routed_expert_read_bytes_per_decode_token == 12_740_198_400
    assert plan.suggested_decode_guard_flags is not None
    assert plan.suggested_decode_guard_flags["decode_read_bytes_per_token"] == (
        12_740_198_400
    )
    assert plan.suggested_decode_guard_flags[
        "decode_max_routed_read_gib_per_token"
    ] == pytest.approx(12_740_198_400 / 1024**3 * 1.05)
    assert plan.suggested_launch_guard_flags is not None
    assert plan.suggested_launch_guard_flags["argv"] == (
        "--require-prepared-memory-profile",
        "--max-live-working-set-mib",
        "8192",
        "--min-free-unified-memory-gib",
        "24",
    )
    assert plan.cache_estimate.bytes_per_token == 95_232
    assert plan.resident_bytes_estimate == estimate_config_resident_bytes(cfg)
    assert plan.resident_bytes_estimate is not None
    assert plan.resident_bytes_estimate > 34 * 1024**3
    assert plan.resident_bytes_estimate_source == "config_estimate"
    assert plan.unified_memory_bytes == 128 * 1024**3
    assert plan.system_reserve_bytes == 24 * 1024**3
    assert plan.runtime_buffer_bytes == 8 * 1024**3
    assert plan.resident_memory_budget_bytes == 96 * 1024**3
    assert plan.resident_memory_pressure_bytes == (
        32 * 1024**3 + plan.resident_bytes_estimate
    )
    assert plan.resident_memory_headroom_bytes == (
        96 * 1024**3 - plan.resident_bytes_estimate
    )
    assert plan.resident_memory_fits_budget is True
    assert plan.decode_cache_bytes_estimate == 99_857_989_632
    assert plan.decode_cache_fits_budget is False
    assert plan.decode_cache_budget_bytes is not None
    assert plan.decode_cache_safe_context_tokens == (
        plan.decode_cache_budget_bytes // plan.cache_estimate.bytes_per_token
    )
    assert plan.decode_cache_safe_context_tokens < 432_960


def test_plan_includes_decode_guard_seconds_from_cold_read_speed() -> None:
    cfg = load_config(FIXTURES / "glm_5_2_config.json")
    plan = build_plan(
        cfg,
        quant_bits=4,
        group_size=64,
        unified_memory_bytes=128 * 1024**3,
        cold_read_gib_per_second=16.0,
    )

    suggested = plan.suggested_decode_guard_flags
    assert suggested is not None
    assert suggested["source"] == "plan"
    assert suggested["prefill_ssd_read_gib_per_second"] == 16.0
    assert suggested["decode_read_seconds_per_token"] == pytest.approx(
        plan.routed_expert_read_bytes_per_decode_token / (16.0 * 1024**3)
    )
    assert suggested["decode_max_routed_read_seconds_per_token"] == pytest.approx(
        plan.routed_expert_read_bytes_per_decode_token / (16.0 * 1024**3) * 1.05
    )
    assert "--prefill-ssd-read-gib-s" in suggested["argv"]
    assert "--decode-max-routed-read-seconds-per-token" in suggested["argv"]


def test_mlp_layer_types_length_is_validated_at_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 3,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "mlp_layer_types": ["dense", "sparse"]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="mlp_layer_types"):
        load_config(cfg_path)


def test_mlp_layer_types_are_normalized_at_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 4,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "mlp_layer_types": ["dense", "mlp", "moe", "moe_sparse"]
}
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.mlp_layer_types == ("dense", "dense", "sparse", "sparse")
    assert cfg.moe_layers == [2, 3]


def test_mlp_layer_types_reject_unknown_values(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "mlp_layer_types": ["dense", "sparce"]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="invalid config mlp_layer_type"):
        load_config(cfg_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("hidden_size", 0, "hidden_size must be positive"),
        ("vocab_size", 0, "vocab_size must be positive"),
        ("moe_layer_freq", 0, "moe_layer_freq must be positive"),
        ("first_k_dense_replace", -1, "first_k_dense_replace must be non-negative"),
    ),
)
def test_config_rejects_invalid_scalar_values(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    payload = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 128,
        "num_hidden_layers": 3,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
    }
    payload[field] = value
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(cfg_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("hidden_size", True, "hidden_size must be an integer"),
        ("vocab_size", True, "vocab_size must be an integer"),
        ("num_hidden_layers", 1.5, "num_hidden_layers must be an integer"),
        ("rms_norm_eps", True, "rms_norm_eps must be numeric"),
        ("rope_theta", "10000", "rope_theta must be numeric"),
        ("dtype", 1, "dtype must be a string"),
        (
            "dtype",
            "float128",
            "dtype must be one of bfloat16, float16, float32, or float64",
        ),
        ("torch_dtype", True, "dtype must be a string"),
        ("model_type", 1, "model_type must be a string"),
        ("model_type", "", "model_type must be non-empty"),
        (
            "eos_token_id",
            True,
            "eos_token_id must be an integer or a list of integers",
        ),
        ("eos_token_id", -1, "eos_token_id must be non-negative"),
        (
            "eos_token_id",
            [1, False],
            "eos_token_id entries must be integers",
        ),
        (
            "eos_token_id",
            [1, -2],
            "eos_token_id entries must be non-negative",
        ),
        ("indexer_rope_interleave", 1, "indexer_rope_interleave must be a boolean"),
        ("rope_interleave", 1, "rope_interleave must be a boolean"),
        ("scoring_func", 1, "scoring_func must be a string"),
        ("scoring_func", "bad", "scoring_func must be sigmoid, softmax, or raw"),
        ("topk_method", 1, "topk_method must be a string"),
        ("norm_topk_prob", 1, "norm_topk_prob must be a boolean"),
        ("routed_scaling_factor", True, "routed_scaling_factor must be numeric"),
        ("routed_scaling_factor", 0, "routed_scaling_factor must be positive"),
        ("n_group", 0, "n_group must be positive"),
        ("topk_group", 0, "topk_group must be positive"),
        ("index_topk_freq", True, "index_topk_freq must be an integer"),
        ("index_topk_freq", 1.5, "index_topk_freq must be an integer"),
        ("index_topk_freq", 0, "index_topk_freq must be positive"),
        ("index_skip_topk_offset", 1.5, "index_skip_topk_offset must be an integer"),
        ("index_skip_topk_offset", -1, "index_skip_topk_offset must be non-negative"),
    ),
)
def test_config_rejects_invalid_scalar_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 128,
        "num_hidden_layers": 3,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
    }
    if field in {"index_topk_freq", "index_skip_topk_offset"}:
        payload["index_topk_freq"] = 1
        payload["index_skip_topk_offset"] = 2
    payload[field] = value
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(cfg_path)


def test_config_rejects_non_object_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ConfigError, match="config JSON must be an object"):
        load_config(cfg_path)


def test_config_resident_estimate_uses_typed_weight_dtype(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "glm_moe_dsa_config.json").read_text())
    payload.pop("n_routed_experts", None)
    payload["dtype"] = "float32"
    float32_path = tmp_path / "float32.json"
    float32_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg32 = load_config(float32_path)
    payload["dtype"] = "torch.bfloat16"
    bf16_path = tmp_path / "bf16.json"
    bf16_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg16 = load_config(bf16_path)

    assert cfg32.weight_dtype == "float32"
    assert cfg32.weight_dtype_bytes == 4
    assert cfg16.weight_dtype == "bfloat16"
    assert cfg16.weight_dtype_bytes == 2
    assert estimate_config_resident_bytes(cfg32) == (
        2 * estimate_config_resident_bytes(cfg16)
    )


def test_config_rejects_experts_per_token_above_expert_count(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 3,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 5
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="num_experts_per_tok"):
        load_config(cfg_path)


def test_config_rejects_topk_group_above_group_count(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 3,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "n_group": 2,
  "topk_group": 3
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="topk_group must not exceed n_group"):
        load_config(cfg_path)


def test_config_rejects_first_k_dense_replace_above_layer_count(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 3,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "first_k_dense_replace": 4
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="first_k_dense_replace"):
        load_config(cfg_path)


def test_config_rejects_index_topk_above_model_context(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 3,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "max_position_embeddings": 128,
  "index_topk": 256
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="index_topk"):
        load_config(cfg_path)


def test_indexer_types_length_is_validated_at_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 3,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "indexer_types": ["full", "shared"]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="indexer_types length"):
        load_config(cfg_path)


def test_index_topk_pattern_length_is_validated_at_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 3,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "index_topk_pattern": "FS"
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="index_topk_pattern length"):
        load_config(cfg_path)


def test_indexer_types_are_normalized_at_load(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 3,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "q_lora_rank": 16,
  "index_head_dim": 8,
  "index_n_heads": 2,
  "index_topk": 4,
  "indexer_types": ["Full", "S", "none"]
}
""",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.indexer_types == ("full", "shared", "none")


@pytest.mark.parametrize(
    "missing_field",
    ("index_head_dim", "index_n_heads", "index_topk", "q_lora_rank"),
)
def test_dsa_indexer_schedule_requires_indexer_dimensions(
    tmp_path: Path,
    missing_field: str,
) -> None:
    payload = {
        "model_type": "glm_moe_dsa",
        "hidden_size": 128,
        "num_hidden_layers": 3,
        "intermediate_size": 256,
        "moe_intermediate_size": 64,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "q_lora_rank": 16,
        "index_head_dim": 8,
        "index_n_heads": 2,
        "index_topk": 4,
        "indexer_types": ["full", "shared", "none"],
    }
    del payload[missing_field]
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match=missing_field):
        load_config(cfg_path)


def test_indexer_types_reject_unknown_values(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "index_topk_pattern": "FX"
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="invalid config DSA indexer type"):
        load_config(cfg_path)


def test_indexer_types_reject_shared_before_full(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        """{
  "model_type": "glm_moe_dsa",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "intermediate_size": 256,
  "moe_intermediate_size": 64,
  "n_routed_experts": 4,
  "num_experts_per_tok": 2,
  "indexer_types": ["shared", "full"]
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="shared DSA indexer"):
        load_config(cfg_path)


def test_plan_cli_writes_prepare_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    flags_path = tmp_path / "prepare-flags.json"

    status = cli_main(
        [
            "plan",
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--unified-memory-gib",
            "128",
            "--cold-read-gib-s",
            "16",
            "--write-prepare-flags",
            str(flags_path),
            "--json",
        ]
    )

    assert status == 0
    capsys.readouterr()
    payload = json.loads(flags_path.read_text(encoding="utf-8"))
    assert payload["source"] == "plan"
    assert payload["argv_safe_to_replay"] is True
    assert "--auto-context-from-budget" in payload["argv"]
    assert "--max-cache-gib" in payload["argv"]
    assert "--unified-memory-gib" in payload["argv"]
    assert "--cold-read-gib-s" in payload["argv"]


def test_plan_cli_rejects_writing_missing_prepare_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    flags_path = tmp_path / "prepare-flags.json"

    status = cli_main(
        [
            "plan",
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--max-cache-gib",
            "0",
            "--write-prepare-flags",
            str(flags_path),
        ]
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "no suggested prepare flags are available to write" in captured.err
    assert not flags_path.exists()


def test_routed_expert_classification_keeps_router_resident() -> None:
    assert is_routed_expert_tensor("model.layers.3.mlp.experts.7.gate_proj.weight")
    assert is_routed_expert_tensor("model.layers.3.mlp.experts.gate_proj.weight")
    assert is_routed_expert_tensor("model.layers.3.mlp.experts.w1.weight")
    assert is_routed_expert_tensor("model.layers.3.mlp.experts.w2.weight")
    assert is_routed_expert_tensor("model.layers.3.mlp.experts.w3.weight")
    assert is_routed_expert_tensor("model.layers.3.mlp.switch_mlp.down_proj.scales")
    assert is_routed_expert_tensor("model.layers.3.mlp.switch_mlp.w2.scales")
    assert not is_routed_expert_tensor("model.layers.3.mlp.gate.weight")
    assert not is_routed_expert_tensor("model.layers.3.mlp.shared_expert.gate_proj.weight")
    assert categorize_tensor("model.layers.3.mlp.gate.weight") == "routers"
