#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_moe_infer_real_attn_projections_smoke import (  # noqa: E402
    mxfp4_dims,
    tensor_map,
    vector_dim,
    write_input,
)
from glm_moe_infer_real_decode_layers_smoke import (  # noqa: E402
    compare_outputs,
    expert_layer_ids,
    run_command,
    write_f32_cache,
)
from largerlm.embedding import embed_token  # noqa: E402


def cache_position_rows_written(cache_layout: Path, cache_file: Path, position: int) -> int:
    layout = json.loads(cache_layout.read_text(encoding="utf-8"))
    raw = cache_file.read_bytes()
    written = 0
    for segment in layout.get("segments", []):
        offset = int(segment["offset"])
        stride = int(segment["token_stride_bytes"])
        width = int(segment["width"])
        dtype_bytes = int(segment["dtype_bytes"])
        row_bytes = width * dtype_bytes
        start = offset + position * stride
        row = raw[start : start + row_bytes]
        if len(row) != row_bytes:
            raise SystemExit("cache row read went out of bounds")
        if any(row):
            written += 1
    return written


def derive_layers_and_dims(prepared: Path, all_layers: bool, layers_csv: str, dense_csv: str):
    resident_layout = prepared / "resident" / "layout.json"
    expert_layout = prepared / "experts" / "layout.json"
    if not resident_layout.exists() or not expert_layout.exists():
        print(f"prepared package missing, skipping: {prepared}")
        return None
    if all_layers:
        expert_layers = expert_layer_ids(expert_layout)
        if not expert_layers:
            raise SystemExit("expert layout has no layers")
        layers = list(range(max(expert_layers) + 1))
        expert_set = set(expert_layers)
        dense_layers = [layer for layer in layers if layer not in expert_set]
    else:
        layers = [int(item) for item in layers_csv.split(",") if item]
        dense_layers = [int(item) for item in dense_csv.split(",") if item]
    if not layers:
        raise SystemExit("--layers must not be empty")
    tensors = tensor_map(resident_layout)
    prefix = f"model.layers.{layers[0]}"
    q_a_out, hidden_dim, _ = mxfp4_dims(tensors, f"{prefix}.self_attn.q_a_proj.weight")
    q_b_out, q_b_in, _ = mxfp4_dims(tensors, f"{prefix}.self_attn.q_b_proj.weight")
    kv_a_out, kv_a_in, _ = mxfp4_dims(
        tensors,
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
    )
    kv_lora_dim = vector_dim(tensors, f"{prefix}.self_attn.kv_a_layernorm.weight")
    kv_rope_dim = kv_a_out - kv_lora_dim
    embed_shape = tensors[f"{prefix}.self_attn.embed_q.weight"]["shape"]
    unembed_shape = tensors[f"{prefix}.self_attn.unembed_out.weight"]["shape"]
    num_heads = int(embed_shape[0])
    qk_nope_dim = int(embed_shape[2]) * 8
    v_head_dim = int(unembed_shape[1])
    if (
        q_b_in != q_a_out
        or kv_a_in != hidden_dim
        or int(embed_shape[1]) != kv_lora_dim
        or int(unembed_shape[0]) != num_heads
        or int(unembed_shape[2]) * 8 != kv_lora_dim
        or q_b_out != num_heads * (qk_nope_dim + kv_rope_dim)
    ):
        raise SystemExit("real GLM decode-layer dims are inconsistent")
    return {
        "resident_layout": resident_layout,
        "layers": layers,
        "dense_layers": dense_layers,
        "hidden_dim": hidden_dim,
        "kv_a_out": kv_a_out,
        "kv_lora_dim": kv_lora_dim,
        "kv_rope_dim": kv_rope_dim,
        "num_heads": num_heads,
        "qk_nope_dim": qk_nope_dim,
        "v_head_dim": v_head_dim,
    }


def run_decode_step(
    *,
    args: argparse.Namespace,
    dims: dict,
    root: Path,
    cache_layout: Path,
    cache_file: Path,
    step: int,
    input_path: Path,
) -> dict:
    output = root / f"step{step}_out.f32"
    topk = root / f"step{step}_topk.json"
    token = root / f"step{step}_token.json"
    next_input = root / f"step{step}_next_input.f32"
    expected_next = root / f"step{step}_expected_next_input.f32"
    work = root / f"step{step}_work"
    cmd = [
        str(args.binary),
        "--prepared",
        str(args.prepared),
        "--probe-decode-layers",
        "--decode-layers",
        ",".join(str(x) for x in dims["layers"]),
        "--input-f32",
        str(input_path),
        "--output-f32",
        str(output),
        "--output-dir",
        str(work),
        "--output-topk-json",
        str(topk),
        "--output-token-json",
        str(token),
        "--output-next-input-f32",
        str(next_input),
        "--cache-layout",
        str(cache_layout),
        "--cache-file",
        str(cache_file),
        "--position",
        str(step),
        "--context-length",
        str(step + 1),
        "--num-heads",
        str(dims["num_heads"]),
        "--kv-lora-dim",
        str(dims["kv_lora_dim"]),
        "--qk-nope-dim",
        str(dims["qk_nope_dim"]),
        "--rope-dim",
        str(dims["kv_rope_dim"]),
        "--v-head-dim",
        str(dims["v_head_dim"]),
        "--cache-position-offset",
        "0",
        "--top-k",
        str(args.top_k),
        "--expert-buffer-count",
        str(args.top_k + 1),
        "--rms-norm-eps",
        f"{args.rms_norm_eps:.9g}",
        "--rope-theta",
        f"{args.rope_theta:.9g}",
        "--max-cache-file-mib",
        "1",
        "--max-cache-read-mib",
        "1",
        "--max-chunk-mib",
        "16",
        "--max-embedding-row-mib",
        "1",
        "--max-live-working-set-mib",
        str(args.max_live_working_set_mib),
        "--skip-debug-intermediates",
        "--include-shared-expert",
        "--json",
    ]
    completed = run_command(cmd, echo=not args.quiet_commands)
    payload = json.loads(completed.stdout)
    decode = payload.get("probe_decode_layers") or {}
    logits = payload.get("probe_final_logits") or {}
    if not payload.get("ok") or not decode.get("ok") or not logits.get("ok"):
        raise SystemExit(f"two-step decode step {step} did not report ok: {payload}")
    if payload.get("estimated_live_working_set_bytes", 0) > args.max_live_working_set_mib * 1024 * 1024:
        raise SystemExit("two-step decode exceeded live working-set cap")
    if decode.get("memory_input_layer_count") != max(0, len(dims["layers"]) - 1):
        raise SystemExit("two-step decode memory chain count mismatch")
    if decode.get("debug_intermediates_written"):
        raise SystemExit("two-step decode wrote debug intermediates")
    debug_files = [p for p in work.rglob("*") if p.is_file()]
    if debug_files:
        raise SystemExit(f"two-step decode wrote unexpected debug files: {debug_files[:3]}")
    generated = logits.get("generated_token") or {}
    topk_items = logits.get("topk") or []
    if not topk_items or generated.get("token_id") != topk_items[0].get("token_id"):
        raise SystemExit("two-step generated token does not match top-1")
    next_embedding = logits.get("next_input_embedding") or {}
    if not next_embedding.get("ok"):
        raise SystemExit("two-step generated-token embedding did not report ok")
    embed_token(
        dims["resident_layout"],
        token_id=int(generated["token_id"]),
        output_f32_path=expected_next,
        max_row_bytes=1024 * 1024,
        expected_hidden_size=int(dims["hidden_dim"]),
    )
    embedding_diff = compare_outputs(expected_next, next_input)
    if embedding_diff["max_abs_diff"] != 0.0:
        raise SystemExit("two-step next embedding differs from Python oracle")
    return {
        "payload": payload,
        "generated_token_id": int(generated["token_id"]),
        "next_input": next_input,
        "embedding_diff": embedding_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run two consecutive glm_moe_infer decode steps with generated-token embedding feedback."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--dense-layers", default="0")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--quiet-commands", action="store_true")
    parser.add_argument("--max-live-working-set-mib", type=int, default=768)
    args = parser.parse_args()

    dims = derive_layers_and_dims(args.prepared, args.all_layers, args.layers, args.dense_layers)
    if dims is None:
        return 0
    root = args.root or Path(
        tempfile.mkdtemp(prefix="largerlm-glm-two-step-real-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    input0 = root / "step0_input.f32"
    write_input(input0, int(dims["hidden_dim"]))
    cache_layout, _, cache_file = write_f32_cache(
        root,
        list(dims["layers"]),
        int(dims["kv_a_out"]),
        max_context_tokens=2,
    )
    print(f"fixture: {root}")
    step0 = run_decode_step(
        args=args,
        dims=dims,
        root=root,
        cache_layout=cache_layout,
        cache_file=cache_file,
        step=0,
        input_path=input0,
    )
    rows0 = cache_position_rows_written(cache_layout, cache_file, 0)
    if rows0 != len(dims["layers"]):
        raise SystemExit(f"cache position 0 rows written {rows0} != {len(dims['layers'])}")
    step1 = run_decode_step(
        args=args,
        dims=dims,
        root=root,
        cache_layout=cache_layout,
        cache_file=cache_file,
        step=1,
        input_path=step0["next_input"],
    )
    rows1 = cache_position_rows_written(cache_layout, cache_file, 1)
    if rows1 != len(dims["layers"]):
        raise SystemExit(f"cache position 1 rows written {rows1} != {len(dims['layers'])}")
    summary = {
        "layers": dims["layers"],
        "dense_layers": dims["dense_layers"],
        "generated_token_ids": [
            step0["generated_token_id"],
            step1["generated_token_id"],
        ],
        "cache_position_rows_written": [rows0, rows1],
        "step_elapsed_seconds": [
            step0["payload"]["probe_decode_layers"]["elapsed_seconds"],
            step1["payload"]["probe_decode_layers"]["elapsed_seconds"],
        ],
        "step_expert_read_seconds": [
            step0["payload"]["probe_decode_layers"]["expert_read_seconds"],
            step1["payload"]["probe_decode_layers"]["expert_read_seconds"],
        ],
        "step_moe_mlp_kernel_seconds": [
            step0["payload"]["probe_decode_layers"]["moe_mlp_kernel_seconds"],
            step1["payload"]["probe_decode_layers"]["moe_mlp_kernel_seconds"],
        ],
        "step_moe_mlp_output_write_seconds": [
            step0["payload"]["probe_decode_layers"]["moe_mlp_output_write_seconds"],
            step1["payload"]["probe_decode_layers"]["moe_mlp_output_write_seconds"],
        ],
        "step_moe_mlp_overhead_seconds": [
            step0["payload"]["probe_decode_layers"]["moe_mlp_overhead_seconds"],
            step1["payload"]["probe_decode_layers"]["moe_mlp_overhead_seconds"],
        ],
        "final_logits_elapsed_seconds": [
            step0["payload"]["probe_final_logits"]["elapsed_seconds"],
            step1["payload"]["probe_final_logits"]["elapsed_seconds"],
        ],
        "estimated_live_working_set_bytes": [
            step0["payload"]["estimated_live_working_set_bytes"],
            step1["payload"]["estimated_live_working_set_bytes"],
        ],
        "next_input_embedding_max_abs_diff": [
            step0["embedding_diff"]["max_abs_diff"],
            step1["embedding_diff"]["max_abs_diff"],
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("  real two-step decode smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
