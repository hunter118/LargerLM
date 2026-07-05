#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _prompt_tokens(value: str) -> list[int]:
    result: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        result.append(_positive_int(item))
    if not result:
        raise argparse.ArgumentTypeError("at least one prompt token count is required")
    return result


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bytes_to_gib(value: Any) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return number / float(1024**3)


def _stderr_tail(stderr: str, *, limit: int = 4000) -> str:
    if len(stderr) <= limit:
        return stderr
    return stderr[-limit:]


def _json_from_stdout(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _chunk_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    auto = _as_dict(plan.get("auto"))
    max_safe = _as_dict(plan.get("max_safe"))
    selected = max_safe if max_safe else auto
    return {
        "configured_is_auto": plan.get("configured_is_auto"),
        "auto_chunk_tokens": _as_int(auto.get("chunk_tokens")),
        "max_safe_chunk_tokens": _as_int(max_safe.get("chunk_tokens")),
        "max_safe_limiting_cap_names": max_safe.get("limiting_cap_names"),
        "selected_mpp_candidate_reachable_under_caps": selected.get(
            "mpp_tensor_ops_candidate_reachable_under_caps"
        ),
        "selected_mpp_candidate_blocking_cap_summary": selected.get(
            "mpp_tensor_ops_candidate_blocking_cap_summary"
        ),
        "selected_mpp_dimension_candidate_matrix_count": _as_int(
            selected.get("mpp_tensor_ops_dimension_candidate_matrix_count")
        ),
        "selected_mpp_stage_tiling_counterfactual_reachable": selected.get(
            "mpp_tensor_ops_candidate_stage_tiling_counterfactual_reachable"
        ),
        "selected_mpp_stage_tiling_counterfactual_blockers": selected.get(
            "mpp_tensor_ops_candidate_stage_tiling_counterfactual_blockers"
        ),
    }


def summarize_payload(
    *,
    prompt_tokens: int,
    returncode: int,
    payload: dict[str, Any] | None,
    stderr: str,
) -> dict[str, Any]:
    if payload is None:
        return {
            "prompt_tokens": prompt_tokens,
            "exit_code": returncode,
            "admitted": False,
            "json_ok": False,
            "error": "inspect-prepared did not return a JSON object",
            "stderr_tail": _stderr_tail(stderr),
        }

    request = _as_dict(payload.get("request_check"))
    prompt_chunk = _as_dict(request.get("prefill_prompt_chunk_tokens"))
    runtime = _as_dict(request.get("runtime_preflight"))
    coverage = _as_dict(request.get("prefill_acceleration_coverage"))
    routed = _as_dict(request.get("prefill_routed_expert_read"))
    stage = _as_dict(request.get("prefill_routed_stage_temp_disk"))
    chunk_plan = _as_dict(request.get("prefill_prompt_chunk_plan"))
    acceleration = _as_dict(payload.get("prefill_backend"))
    capability = _as_dict(acceleration.get("capability"))
    neural = _as_dict(capability.get("prefill_neural_accelerator_status"))

    request_ok = request.get("ok") is True
    runtime_ok = runtime.get("available_memory_ok") is True if runtime else None
    admitted = returncode == 0 and request_ok and runtime_ok is not False
    error = payload.get("error") or request.get("error")
    if error is None and returncode != 0:
        error = "inspect-prepared exited non-zero"

    return {
        "prompt_tokens": prompt_tokens,
        "exit_code": returncode,
        "admitted": admitted,
        "json_ok": True,
        "request_ok": request_ok,
        "error": error,
        "prompt_chunk_tokens": {
            "configured": prompt_chunk.get("configured"),
            "resolved": prompt_chunk.get("resolved"),
            "max_safe": prompt_chunk.get("max_safe"),
        },
        "runtime_preflight": {
            "ran": runtime.get("ran"),
            "available_memory_ok": runtime.get("available_memory_ok"),
            "required_available_gib": _bytes_to_gib(
                runtime.get("required_available_memory_bytes")
            ),
            "system_available_gib": _bytes_to_gib(
                runtime.get("system_available_memory_bytes")
            ),
            "live_working_set_gib": _bytes_to_gib(
                runtime.get("live_working_set_bytes")
            ),
            "max_live_working_set_gib": _bytes_to_gib(
                runtime.get("max_live_working_set_bytes")
            ),
            "system_memory_source": runtime.get("system_memory_source"),
        },
        "prefill_acceleration": {
            "ok": coverage.get("ok"),
            "required": coverage.get("required"),
            "accelerated_backends": coverage.get("accelerated_backends"),
            "accelerated_matrix_count": _as_int(
                coverage.get("accelerated_matrix_count")
            ),
            "accelerated_flop_fraction": _as_float(
                coverage.get("accelerated_flop_fraction")
            ),
            "mpsgraph_matrix_count": _as_int(coverage.get("mpsgraph_matrix_count")),
            "unsupported_mpsgraph_matrix_count": _as_int(
                coverage.get("unsupported_mpsgraph_matrix_count")
            ),
            "mpp_candidate_matrix_count": _as_int(
                coverage.get("mpp_tensor_ops_candidate_matrix_count")
            ),
            "mpp_candidate_flop_fraction": _as_float(
                coverage.get("mpp_tensor_ops_candidate_flop_fraction")
            ),
            "streamed_routed_expert_mpp_candidate_matrix_count": _as_int(
                coverage.get("streamed_routed_expert_mpp_candidate_matrix_count")
            ),
            "accelerated_router_gate_only": coverage.get(
                "accelerated_router_gate_only"
            ),
            "reason": coverage.get("reason"),
        },
        "backend_probe": {
            "selectable_accelerated_prefill_backends": capability.get(
                "selectable_accelerated_prefill_backends"
            ),
            "validated_accelerated_prefill_backends": capability.get(
                "validated_accelerated_prefill_backends"
            ),
            "mps_graph_probe_ok": capability.get("mps_graph_probe_ok"),
            "mpp_tensor_ops_symbol_declared": capability.get(
                "mpp_tensor_ops_symbol_declared"
            ),
            "neural_accelerator_status": neural.get("status"),
            "neural_accelerator_reason": neural.get("reason"),
        },
        "routed_expert_read": {
            "prompt_chunk_tokens": routed.get("prompt_chunk_tokens"),
            "chunks_per_prompt": routed.get("chunks_per_prompt"),
            "planned_read_gib": _bytes_to_gib(routed.get("planned_read_bytes")),
            "planned_read_seconds": _as_float(routed.get("planned_read_seconds")),
            "read_amplification": _as_float(routed.get("read_amplification")),
            "within_limit": routed.get("within_limit"),
            "within_seconds_limit": routed.get("within_seconds_limit"),
            "minimum_chunk_tokens_for_limits": routed.get(
                "minimum_chunk_tokens_for_limits"
            ),
        },
        "stage_temp": {
            "prompt_chunk_tokens": stage.get("prompt_chunk_tokens"),
            "expert_stage_tiling": stage.get("expert_stage_tiling"),
            "max_stage_gib": _bytes_to_gib(stage.get("max_stage_bytes")),
            "max_compact_stage_gib": _bytes_to_gib(
                stage.get("max_compact_stage_bytes")
            ),
            "max_stage_plus_compact_gib": _bytes_to_gib(
                stage.get("max_stage_plus_compact_bytes")
            ),
            "max_stage_raw_ranges": stage.get("max_stage_raw_ranges"),
            "max_stage_coalesced_ranges": stage.get("max_stage_coalesced_ranges"),
            "within_limit": stage.get("within_limit"),
            "within_stage_raw_range_limit": stage.get(
                "within_stage_raw_range_limit"
            ),
            "within_stage_coalesced_range_limit": stage.get(
                "within_stage_coalesced_range_limit"
            ),
        },
        "prefill_prompt_chunk_plan": _chunk_plan_summary(chunk_plan),
        "stderr_tail": _stderr_tail(stderr) if returncode != 0 else "",
    }


def build_inspect_command(args: argparse.Namespace, prompt_tokens: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "largerlm",
        "inspect-prepared",
        args.prepared,
        "--require-prepared-memory-profile",
        "--require-glm-4bit",
        "--require-public-glm-5-2-shape",
        "--check-prompt-tokens",
        str(prompt_tokens),
        "--check-max-new-tokens",
        str(args.max_new_tokens),
        "--check-runtime-preflight",
        "--json",
    ]
    if args.apply_launch_profile:
        command.extend(["--apply-launch-profile", args.apply_launch_profile])
    if args.lock_launch_profile:
        command.append("--lock-launch-profile")
    if args.require_locked_launch_profile:
        command.append("--require-locked-launch-profile")
    if args.allow_non_accelerated_prefill_launch_audit:
        command.append("--allow-non-accelerated-prefill-launch-audit")
    if args.run_mpsgraph_probe:
        command.append("--run-mpsgraph-probe")
    if args.run_mpp_probe:
        command.append("--run-mpp-probe")
    if args.probe_timeout_seconds is not None:
        command.extend(
            [
                "--prefill-backend-probe-timeout-seconds",
                str(args.probe_timeout_seconds),
            ]
        )
    for item in args.inspect_arg:
        command.append(item)
    return command


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    raw_dir = Path(args.write_raw_dir) if args.write_raw_dir else None
    for prompt_tokens in args.prompt_tokens:
        command = build_inspect_command(args, prompt_tokens)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            cwd=args.cwd,
        )
        payload = _json_from_stdout(completed.stdout)
        if raw_dir is not None and payload is not None:
            _write_json(raw_dir / f"inspect-{prompt_tokens}tok.json", payload)
        entry = summarize_payload(
            prompt_tokens=prompt_tokens,
            returncode=completed.returncode,
            payload=payload,
            stderr=completed.stderr,
        )
        entry["command"] = command
        entries.append(entry)

    admitted = [
        entry["prompt_tokens"]
        for entry in entries
        if entry.get("admitted") is True and isinstance(entry.get("prompt_tokens"), int)
    ]
    rejected = [
        entry["prompt_tokens"]
        for entry in entries
        if entry.get("admitted") is not True and isinstance(entry.get("prompt_tokens"), int)
    ]
    return {
        "schema": "largerlm.long_prompt_readiness_matrix.v1",
        "prepared": args.prepared,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "all_admitted": len(rejected) == 0,
        "max_admitted_prompt_tokens": max(admitted) if admitted else None,
        "first_rejected_prompt_tokens": min(rejected) if rejected else None,
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run inspect-prepared admission checks for several long-prompt sizes "
            "without starting generation or reading model weight payloads."
        )
    )
    parser.add_argument("prepared")
    parser.add_argument(
        "--prompt-tokens",
        type=_prompt_tokens,
        default=[128, 512, 2048, 4096],
        help="comma-separated prompt token counts to inspect",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=1)
    parser.add_argument("--apply-launch-profile")
    parser.add_argument("--lock-launch-profile", action="store_true")
    parser.add_argument("--require-locked-launch-profile", action="store_true")
    parser.add_argument(
        "--allow-non-accelerated-prefill-launch-audit",
        action="store_true",
    )
    parser.add_argument("--run-mpsgraph-probe", action="store_true")
    parser.add_argument("--run-mpp-probe", action="store_true")
    parser.add_argument("--probe-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--inspect-arg",
        action="append",
        default=[],
        help="extra single argv item appended to each inspect-prepared command",
    )
    parser.add_argument("--write-raw-dir")
    parser.add_argument("--write-result")
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = run_matrix(args)
    if args.write_result:
        _write_json(Path(args.write_result), payload)
    if args.json or not args.write_result:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
