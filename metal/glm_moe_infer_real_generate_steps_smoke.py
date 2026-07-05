#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_moe_infer_real_attn_projections_smoke import write_input  # noqa: E402
from glm_moe_infer_real_decode_layers_smoke import (  # noqa: E402
    compare_outputs,
    run_command,
    write_f32_cache,
)
from glm_moe_infer_real_two_step_decode_smoke import (  # noqa: E402
    cache_position_rows_written,
    derive_layers_and_dims,
)
from largerlm.embedding import embed_token  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run glm_moe_infer greedy multi-step decode in one process on real GLM weights."
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
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--quiet-commands", action="store_true")
    parser.add_argument("--max-live-working-set-mib", type=int, default=768)
    parser.add_argument("--min-free-unified-memory-gib", type=float, default=0.0)
    parser.add_argument("--first-from-input-logits", action="store_true")
    parser.add_argument("--input-token-id", type=int, default=None)
    parser.add_argument("--request-json", action="store_true")
    args = parser.parse_args()

    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    dims = derive_layers_and_dims(args.prepared, args.all_layers, args.layers, args.dense_layers)
    if dims is None:
        return 0
    root = args.root or Path(
        tempfile.mkdtemp(prefix="largerlm-glm-generate-steps-real-", dir="/private/tmp")
    )
    root.mkdir(parents=True, exist_ok=True)
    input0 = root / "input.f32"
    output = root / "output.f32"
    generated_json = root / "generated.json"
    final_next = root / "final_next_input.f32"
    expected_final_next = root / "expected_final_next_input.f32"
    work = root / "work"
    if args.input_token_id is None:
        write_input(input0, int(dims["hidden_dim"]))
    cache_layout, _, cache_file = write_f32_cache(
        root,
        list(dims["layers"]),
        int(dims["kv_a_out"]),
        max_context_tokens=args.steps,
    )
    print(f"fixture: {root}")
    decode_layers_csv = ",".join(str(x) for x in dims["layers"])
    request = {
        "decode_layers": decode_layers_csv,
        "output_f32": str(output),
        "output_dir": str(work),
        "output_generated_json": str(generated_json),
        "output_next_input_f32": str(final_next),
        "generate_steps": args.steps,
        "generate_first_from_input_logits": args.first_from_input_logits,
        "cache_layout": str(cache_layout),
        "cache_file": str(cache_file),
        "position": 0,
        "context_length": 1,
        "num_heads": int(dims["num_heads"]),
        "kv_lora_dim": int(dims["kv_lora_dim"]),
        "qk_nope_dim": int(dims["qk_nope_dim"]),
        "rope_dim": int(dims["kv_rope_dim"]),
        "v_head_dim": int(dims["v_head_dim"]),
        "cache_position_offset": 0,
        "top_k": args.top_k,
        "expert_buffer_count": args.top_k + 1,
        "rms_norm_eps": args.rms_norm_eps,
        "rope_theta": args.rope_theta,
        "max_cache_file_mib": 1,
        "max_cache_read_mib": 1,
        "max_chunk_mib": 16,
        "max_embedding_row_mib": 1,
        "max_live_working_set_mib": args.max_live_working_set_mib,
        "min_free_unified_memory_gib": args.min_free_unified_memory_gib,
        "skip_debug_intermediates": True,
        "include_shared_expert": True,
    }
    if args.input_token_id is None:
        request["input_f32"] = str(input0)
    else:
        request["input_token_id"] = args.input_token_id

    if args.request_json:
        request_path = root / "generate_request.json"
        request_path.write_text(
            json.dumps(request, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        cmd = [
            str(args.binary),
            "--prepared",
            str(args.prepared),
            "--generate-request-json",
            str(request_path),
            "--json",
        ]
    else:
        cmd = [
            str(args.binary),
            "--prepared",
            str(args.prepared),
            "--generate-token-ids",
            "--decode-layers",
            request["decode_layers"],
        ]
        if args.input_token_id is None:
            cmd += ["--input-f32", request["input_f32"]]
        else:
            cmd += ["--input-token-id", str(request["input_token_id"])]
        cmd += [
            "--output-f32",
            request["output_f32"],
            "--output-dir",
            request["output_dir"],
            "--output-generated-json",
            request["output_generated_json"],
            "--output-next-input-f32",
            request["output_next_input_f32"],
            "--generate-steps",
            str(request["generate_steps"]),
        ]
        if args.first_from_input_logits:
            cmd.append("--generate-first-from-input-logits")
        cmd += [
            "--cache-layout",
            request["cache_layout"],
            "--cache-file",
            request["cache_file"],
            "--position",
            str(request["position"]),
            "--context-length",
            str(request["context_length"]),
            "--num-heads",
            str(request["num_heads"]),
            "--kv-lora-dim",
            str(request["kv_lora_dim"]),
            "--qk-nope-dim",
            str(request["qk_nope_dim"]),
            "--rope-dim",
            str(request["rope_dim"]),
            "--v-head-dim",
            str(request["v_head_dim"]),
            "--cache-position-offset",
            str(request["cache_position_offset"]),
            "--top-k",
            str(request["top_k"]),
            "--expert-buffer-count",
            str(request["expert_buffer_count"]),
            "--rms-norm-eps",
            f"{request['rms_norm_eps']:.9g}",
            "--rope-theta",
            f"{request['rope_theta']:.9g}",
            "--max-cache-file-mib",
            str(request["max_cache_file_mib"]),
            "--max-cache-read-mib",
            str(request["max_cache_read_mib"]),
            "--max-chunk-mib",
            str(request["max_chunk_mib"]),
            "--max-embedding-row-mib",
            str(request["max_embedding_row_mib"]),
            "--max-live-working-set-mib",
            str(request["max_live_working_set_mib"]),
            "--min-free-unified-memory-gib",
            f"{request['min_free_unified_memory_gib']:.9g}",
            "--skip-debug-intermediates",
            "--include-shared-expert",
            "--json",
        ]
    completed = run_command(cmd, echo=not args.quiet_commands)
    payload = json.loads(completed.stdout)
    generate = payload.get("probe_generate") or {}
    decode = payload.get("probe_decode_layers") or {}
    logits = payload.get("probe_final_logits") or {}
    if not payload.get("ok") or not decode.get("ok") or not logits.get("ok") or not generate.get("ok"):
        raise SystemExit(f"generate-steps probe did not report ok: {payload}")
    if not payload.get("admission_ok"):
        raise SystemExit("glm_moe_infer did not report admission_ok for normal smoke")
    if not payload.get("live_working_set_ok"):
        raise SystemExit("glm_moe_infer did not report live_working_set_ok")
    if not payload.get("available_unified_memory_ok"):
        raise SystemExit("glm_moe_infer did not report available_unified_memory_ok")
    if payload.get("expert_buffer_count_allocated") != payload.get("expert_buffer_count"):
        raise SystemExit("glm_moe_infer did not allocate the planned expert buffers")
    if payload.get("runtime_entry") != "generate_token_ids":
        raise SystemExit("glm_moe_infer did not report generate_token_ids runtime entry")
    if generate.get("entry") != "generate_token_ids":
        raise SystemExit("probe_generate did not report generate_token_ids entry")
    if generate.get("step_count") != args.steps:
        raise SystemExit("generate-steps count mismatch")
    steps = generate.get("steps") or []
    if len(steps) != args.steps:
        raise SystemExit("generate-steps payload length mismatch")
    if bool(generate.get("first_step_from_input_logits")) != args.first_from_input_logits:
        raise SystemExit("generate-steps first-from-input-logits mode mismatch")
    if args.input_token_id is None:
        if steps[0].get("input_from_memory"):
            raise SystemExit("generate step 0 should read the fixture input")
    elif not steps[0].get("input_from_memory"):
        raise SystemExit("generate step 0 should use runtime-decoded token embedding")
    if any(not item.get("input_from_memory") for item in steps[1:]):
        raise SystemExit("generate later steps should use in-memory embeddings")
    input_token_embedding = payload.get("probe_input_token_embedding")
    if args.input_token_id is None:
        if input_token_embedding:
            raise SystemExit("unexpected input-token embedding payload")
    else:
        if not input_token_embedding or not input_token_embedding.get("ok"):
            raise SystemExit("missing input-token embedding payload")
        if input_token_embedding.get("token_id") != args.input_token_id:
            raise SystemExit("input-token embedding payload used the wrong token")
    if args.first_from_input_logits:
        if steps[0].get("decode_ran"):
            raise SystemExit("first-from-input-logits step 0 should skip decode")
        if any(not item.get("decode_ran") for item in steps[1:]):
            raise SystemExit("first-from-input-logits later steps should run decode")
    elif any(not item.get("decode_ran", True) for item in steps):
        raise SystemExit("normal generate steps should all run decode")
    generated_ids = [int(item) for item in generate["generated_token_ids"]]
    generated_file = json.loads(generated_json.read_text(encoding="utf-8"))
    file_ids = [
        int(item["generated_token"]["token_id"])
        for item in generated_file.get("steps", [])
    ]
    if file_ids != generated_ids:
        raise SystemExit("generated JSON file does not match stdout payload")
    expected_cache_positions = args.steps - 1 if args.first_from_input_logits else args.steps
    for position in range(expected_cache_positions):
        rows = cache_position_rows_written(cache_layout, cache_file, position)
        if rows != len(dims["layers"]):
            raise SystemExit(
                f"cache position {position} rows written {rows} != {len(dims['layers'])}"
            )
    if not final_next.exists():
        raise SystemExit("generate-steps did not write final next input")
    embed_token(
        dims["resident_layout"],
        token_id=generated_ids[-1],
        output_f32_path=expected_final_next,
        max_row_bytes=1024 * 1024,
        expected_hidden_size=int(dims["hidden_dim"]),
    )
    next_diff = compare_outputs(expected_final_next, final_next)
    if next_diff["max_abs_diff"] != 0.0:
        raise SystemExit("final generated-token embedding differs from Python oracle")
    summary = {
        "layers": dims["layers"],
        "dense_layers": dims["dense_layers"],
        "generated_token_ids": generated_ids,
        "input_token_id": args.input_token_id,
        "request_json": args.request_json,
        "input_token_embedding_bytes_read": (
            0 if args.input_token_id is None else input_token_embedding["bytes_read"]
        ),
        "cache_position_rows_written": [
            cache_position_rows_written(cache_layout, cache_file, position)
            for position in range(expected_cache_positions)
        ],
        "decode_elapsed_seconds": [
            float(item["decode_elapsed_seconds"]) for item in steps
        ],
        "final_logits_elapsed_seconds": [
            float(item["final_logits_elapsed_seconds"]) for item in steps
        ],
        "estimated_live_working_set_bytes": payload[
            "estimated_live_working_set_bytes"
        ],
        "admission_ok": payload["admission_ok"],
        "available_unified_memory_ok": payload["available_unified_memory_ok"],
        "min_free_unified_memory_gib": payload["min_free_unified_memory_gib"],
        "required_available_memory_bytes": payload[
            "required_available_memory_bytes"
        ],
        "system_available_memory_bytes": payload["system_available_memory_bytes"],
        "expert_buffer_count_allocated": payload["expert_buffer_count_allocated"],
        "final_next_input_max_abs_diff": next_diff["max_abs_diff"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("  real generate-steps smoke result: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
