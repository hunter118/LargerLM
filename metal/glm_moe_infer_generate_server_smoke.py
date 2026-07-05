#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm_moe_infer_real_decode_layers_smoke import write_f32_cache  # noqa: E402
from glm_moe_infer_real_two_step_decode_smoke import derive_layers_and_dims  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke the glm_moe_infer formal generation JSONL service scaffold."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/glm-5.2-mxfp4/largerlm-prepared"),
    )
    parser.add_argument("--binary", type=Path, default=Path("metal/glm_moe_infer"))
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--dense-layers", default="0")
    parser.add_argument("--min-free-unified-memory-gib", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--write-executor-stdout", action="store_true")
    parser.add_argument("--write-output-f32", action="store_true")
    parser.add_argument("--write-generated-files", action="store_true")
    parser.add_argument("--file-decode-cache", action="store_true")
    parser.add_argument(
        "--prompt-token-ids",
        default=None,
        help="comma-separated prompt token ids for runtime prompt prefill mode",
    )
    args = parser.parse_args()
    if args.requests <= 0:
        raise SystemExit("--requests must be positive")
    prompt_token_ids = None
    if args.prompt_token_ids:
        prompt_token_ids = [
            int(item.strip())
            for item in args.prompt_token_ids.split(",")
            if item.strip()
        ]
        if not prompt_token_ids:
            raise SystemExit("--prompt-token-ids must include at least one token")
        if any(item < 0 for item in prompt_token_ids):
            raise SystemExit("--prompt-token-ids must be non-negative")

    dims = derive_layers_and_dims(
        args.prepared,
        False,
        args.layers,
        args.dense_layers,
    )
    if dims is None:
        return 0
    root = Path(
        tempfile.mkdtemp(prefix="largerlm-glm-generate-server-smoke-", dir="/private/tmp")
    )
    requests = []
    executor_stdout_paths = []
    output_f32_paths = []
    for request_index in range(args.requests):
        request_root = root / f"request-{request_index + 1}"
        request_root.mkdir()
        max_context_tokens = 2
        if prompt_token_ids:
            max_context_tokens = max_context_tokens + len(prompt_token_ids) - 1
        cache_layout, _, cache_file = write_f32_cache(
            request_root,
            list(dims["layers"]),
            int(dims["kv_a_out"]),
            max_context_tokens=max_context_tokens,
        )
        executor_stdout = request_root / "executor_stdout.json"
        request = {
            "decode_layers": ",".join(str(item) for item in dims["layers"]),
            "output_dir": str(request_root / "work"),
            "generate_steps": 2,
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
            "top_k": 8,
            "expert_buffer_count": 9,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_cache_file_mib": 1,
            "max_cache_read_mib": 1,
            "max_chunk_mib": 16,
            "max_embedding_row_mib": 1,
            "max_live_working_set_mib": 768,
            "min_free_unified_memory_gib": args.min_free_unified_memory_gib,
            "skip_debug_intermediates": True,
            "include_shared_expert": True,
            "dry_run": not args.execute,
        }
        if prompt_token_ids:
            request["prompt_token_ids"] = prompt_token_ids
        else:
            request["input_token_id"] = 0
        if args.file_decode_cache:
            request["in_memory_decode_cache"] = False
        output_f32 = request_root / "output.f32"
        if args.write_output_f32:
            request["output_f32"] = str(output_f32)
        if args.write_generated_files:
            request["output_generated_json"] = str(request_root / "generated.json")
            request["output_next_input_f32"] = str(request_root / "next_input.f32")
        if args.execute and args.write_executor_stdout:
            request["executor_stdout_json"] = str(executor_stdout)
        requests.append(request)
        executor_stdout_paths.append(executor_stdout)
        output_f32_paths.append(output_f32)
    stdin = "\n".join(
        json.dumps(request, separators=(",", ":")) for request in requests
    )
    stdin += "\n" + json.dumps({"command": "quit"}, separators=(",", ":")) + "\n"
    completed = subprocess.run(
        [
            str(args.binary),
            "--prepared",
            str(args.prepared),
            "--generate-server-jsonl",
        ],
        input=stdin,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    lines = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    expected_lines = args.requests + 2
    if len(lines) != expected_lines:
        raise SystemExit(
            f"expected ready/{args.requests} request/done JSONL lines, got {len(lines)}"
        )
    ready = lines[0]
    responses = lines[1:-1]
    done = lines[-1]
    if ready.get("event") != "ready" or not ready.get("ok"):
        raise SystemExit(f"server did not report ready: {ready}")
    if not ready.get("execution_ready"):
        raise SystemExit(f"server did not report runtime readiness: {ready}")
    if ready.get("execution_status") != "runtime_context_ready":
        raise SystemExit(f"server did not report runtime_context_ready: {ready}")
    if int(ready.get("expert_files_opened") or 0) <= 0:
        raise SystemExit(f"server did not keep expert files open: {ready}")
    if not ready.get("resident_fd_opened"):
        raise SystemExit(f"server did not keep resident fd open: {ready}")
    if ready.get("decode_cache_fd_opened"):
        raise SystemExit(f"server should not open a decode cache fd before a request: {ready}")
    for response_index, response in enumerate(responses):
        if response.get("event") != "request" or not response.get("accepted"):
            raise SystemExit(f"server did not accept request: {response}")
        if not response.get("runtime_context_ready"):
            raise SystemExit(
                f"server response did not report runtime context readiness: {response}"
            )
        if args.execute:
            if not response.get("execution_ready"):
                raise SystemExit("server execute request should report execution_ready")
            if not response.get("executed"):
                raise SystemExit("server execute request should run decode")
            if not response.get("runtime_reused"):
                raise SystemExit("server execute request should reuse the runtime context")
            if response.get("execution_status") != "executed_once":
                raise SystemExit("server execute request should report executed_once")
            if response.get("executor_exit_code") != 0:
                raise SystemExit(f"executor failed through server: {response}")
            expected_stdout = executor_stdout_paths[response_index]
            if args.write_executor_stdout:
                if response.get("executor_stdout_json") != str(expected_stdout):
                    raise SystemExit("server did not report the executor stdout path")
                executor_payload = json.loads(expected_stdout.read_text(encoding="utf-8"))
            else:
                if response.get("executor_stdout_json") is not None:
                    raise SystemExit("server should not require executor stdout file")
                if not response.get("executor_payload_parsed"):
                    raise SystemExit(f"server did not parse captured executor payload: {response}")
                executor_payload = response
            if int(response.get("expert_buffer_count_runtime_allocated") or 0) < 9:
                raise SystemExit(
                    "server did not keep runtime expert buffers allocated after execute"
                )
            if args.file_decode_cache:
                if response.get("decode_cache_backend") != "file":
                    raise SystemExit("server did not report file decode cache backend")
                if not response.get("decode_cache_fd_opened"):
                    raise SystemExit("server did not keep decode cache fd open after execute")
                if int(response.get("decode_cache_fd_open_count") or 0) < response_index + 1:
                    raise SystemExit("server did not report decode cache fd opens")
            else:
                if response.get("decode_cache_backend") != "memory":
                    raise SystemExit("server did not report memory decode cache backend")
                if response.get("decode_cache_fd_opened"):
                    raise SystemExit("memory decode cache should not keep a cache fd open")
                if not response.get("decode_cache_memory_loaded"):
                    raise SystemExit("server did not keep decode cache in runtime memory")
                if int(response.get("decode_cache_memory_bytes") or 0) <= 0:
                    raise SystemExit("server did not report decode cache memory bytes")
            generate = executor_payload.get("probe_generate") or {}
            decode_payload = executor_payload.get("probe_decode_layers") or {}
            if not executor_payload.get("admission_ok"):
                raise SystemExit("executor payload did not report admission_ok")
            executor_cache_backend = (
                executor_payload.get("decode_cache_backend")
                or executor_payload.get("executor_decode_cache_backend")
            )
            if args.file_decode_cache:
                if executor_cache_backend != "file":
                    raise SystemExit("executor did not report file decode cache backend")
                if not (
                    executor_payload.get("decode_cache_fd_opened") or
                    executor_payload.get("executor_decode_cache_fd_opened")
                ):
                    raise SystemExit("executor payload did not report decode cache fd open")
            else:
                if executor_cache_backend != "memory":
                    raise SystemExit("executor did not report memory decode cache backend")
                if not (
                    executor_payload.get("decode_cache_memory_loaded") or
                    executor_payload.get("executor_decode_cache_memory_loaded")
                ):
                    raise SystemExit("executor payload did not report decode cache memory load")
            if response_index > 0 and not executor_payload.get("expert_buffer_pool_reused"):
                raise SystemExit("second server execute request did not reuse expert buffers")
            if not generate.get("ok"):
                raise SystemExit(f"executor did not report generation ok: {executor_payload}")
            if prompt_token_ids:
                generated = generate.get("generated_token_ids")
                if not isinstance(generated, list) or len(generated) != 2:
                    raise SystemExit(
                        "executor generated an unexpected prompt-prefill token payload: "
                        f"{generated}"
                    )
                prompt_prefill = generate.get("prompt_prefill") or {}
                if not prompt_prefill.get("ok"):
                    raise SystemExit(
                        f"executor did not report prompt prefill ok: {executor_payload}"
                    )
                if prompt_prefill.get("token_count") != len(prompt_token_ids):
                    raise SystemExit("executor reported the wrong prompt token count")
                prefill_steps = prompt_prefill.get("steps")
                if (
                    not isinstance(prefill_steps, list) or
                    len(prefill_steps) != len(prompt_token_ids)
                ):
                    raise SystemExit("executor did not report every prompt prefill step")
                if not generate.get("first_step_from_input_logits"):
                    raise SystemExit("prompt mode should take first logits from prompt state")
                steps = generate.get("steps")
                if not isinstance(steps, list) or len(steps) != 2:
                    raise SystemExit("executor did not report prompt generation steps")
                first_step = steps[0]
                if not isinstance(first_step, dict):
                    raise SystemExit("first prompt generation step payload is not an object")
                if float(first_step.get("decode_elapsed_seconds", -1.0)) != 0.0:
                    raise SystemExit("first prompt generation step should skip decode")
                observed_prefill_tokens = [
                    int(step.get("token_id"))
                    for step in prefill_steps
                    if isinstance(step, dict) and "token_id" in step
                ]
                if observed_prefill_tokens != prompt_token_ids:
                    raise SystemExit("prompt prefill payload did not preserve token ids")
                if response.get("prompt_token_count") != len(prompt_token_ids):
                    raise SystemExit("decode payload did not preserve prompt token count")
            elif generate.get("generated_token_ids") != [30423, 11093]:
                raise SystemExit(
                    "executor generated unexpected tokens: "
                    f"{generate.get('generated_token_ids')}"
                )
            if args.write_generated_files:
                if "output_generated_json" not in generate:
                    raise SystemExit("executor did not report generated JSON path")
            else:
                if "output_generated_json" in generate:
                    raise SystemExit("default server execute should not write generated JSON")
                for step in generate.get("steps", []):
                    if not isinstance(step, dict):
                        raise SystemExit("generate step payload is not an object")
                    embedding = step.get("next_input_embedding") or {}
                    if float(embedding.get("write_seconds", 0.0)) != 0.0:
                        raise SystemExit(
                            "default server execute should keep next input embedding in memory"
                        )
            output_f32 = output_f32_paths[response_index]
            if args.write_output_f32:
                if decode_payload.get("output_f32") != str(output_f32):
                    raise SystemExit("executor did not report final output_f32 path")
                if not output_f32.exists():
                    raise SystemExit("executor did not write final output_f32")
            else:
                if "output_f32" in decode_payload:
                    raise SystemExit("default server execute should not report output_f32")
                if output_f32.exists():
                    raise SystemExit("default server execute should not write output_f32")
        else:
            if response.get("execution_ready"):
                raise SystemExit("server scaffold must report execution_ready=false")
            if response.get("executed"):
                raise SystemExit("server dry-run request should not execute decode")
            if response.get("execution_status") != "request_protocol_only":
                raise SystemExit("server dry-run request should stay protocol-only")
            if response.get("runtime_reused"):
                raise SystemExit("server dry-run request should not execute through the runtime")
        if response.get("runtime_entry") != "generate_token_ids":
            raise SystemExit("server response did not preserve formal generation entry")
        if response.get("generate_steps") != 2:
            raise SystemExit("server response did not preserve generate_steps")
        if response.get("min_free_unified_memory_gib") != args.min_free_unified_memory_gib:
            raise SystemExit("server response did not preserve min-free reserve")
    if done.get("event") != "done" or done.get("requests") != args.requests:
        raise SystemExit(f"server did not report {args.requests} completed requests: {done}")
    print("  generate server smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
