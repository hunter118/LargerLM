#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "largerlm.multi_prompt_replay_plan.v1"


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int_list(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    parsed: list[int] = []
    for item in value:
        if type(item) is not int:
            return None
        parsed.append(item)
    return parsed


def _first_int_list(*values: object) -> list[int] | None:
    for value in values:
        parsed = _int_list(value)
        if parsed is not None:
            return parsed
    return None


def _raw_response(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("raw_response")
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _largerlm_response_token_result(payload: dict[str, Any]) -> dict[str, Any]:
    response = _as_mapping(payload.get("response"))
    largerlm = _as_mapping(response.get("largerlm"))
    token_result = _as_mapping(largerlm.get("token_result"))
    if token_result:
        return token_result
    raw = _raw_response(payload)
    raw_largerlm = _as_mapping(raw.get("largerlm"))
    return _as_mapping(raw_largerlm.get("token_result"))


def _token_result(payload: dict[str, Any]) -> dict[str, Any]:
    direct = _as_mapping(payload.get("token_result"))
    if direct:
        return direct
    nested = _largerlm_response_token_result(payload)
    if nested:
        return nested
    return {}


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    request = _as_mapping(payload.get("request"))
    token_request = _as_mapping(_token_result(payload).get("request"))
    return request or token_request


def extract_prompt_record(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    request = _request(payload)
    token_result = _token_result(payload)
    prompt_token_ids = _first_int_list(
        request.get("prompt_token_ids"),
        token_result.get("prompt_token_ids"),
        payload.get("prompt_token_ids"),
    )
    if prompt_token_ids is None:
        raise SystemExit(f"{path} does not contain prompt_token_ids")
    generated_token_ids = _first_int_list(
        token_result.get("generated_token_ids"),
        payload.get("generated_token_ids"),
    )
    max_new_tokens = request.get("max_new_tokens")
    if type(max_new_tokens) is not int:
        max_tokens = request.get("max_tokens")
        max_new_tokens = max_tokens if type(max_tokens) is int else None
    label = _safe_label(path.stem)
    return {
        "label": label,
        "source_path": str(path),
        "source_sha256": _sha256_file(path),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_count": len(prompt_token_ids),
        "source_generated_token_ids": generated_token_ids,
        "source_max_new_tokens": max_new_tokens,
        "source_schema": payload.get("schema"),
    }


def _safe_label(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("-")
    label = "".join(allowed).strip("-_")
    return label or "prompt"


@dataclass(frozen=True)
class Variant:
    name: str
    prepared: Path
    launch_profile: Path
    launch_audit: Path
    env: dict[str, str]


def _audit_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "ok": False,
            "request_ok": False,
            "safe_to_replay": False,
            "locked": False,
            "matches_prepared": False,
        }
    payload = _load_json(path)
    launch_audit = _as_mapping(payload.get("launch_audit"))
    request_check = _as_mapping(payload.get("request_check"))
    applied = _as_mapping(payload.get("applied_launch_profile"))
    return {
        "path": str(path),
        "exists": True,
        "sha256": _sha256_file(path),
        "ok": bool(launch_audit.get("ok")),
        "request_ok": bool(request_check.get("ok")),
        "safe_to_replay": bool(applied.get("argv_safe_to_replay")),
        "locked": bool(applied.get("locked")),
        "matches_prepared": bool(applied.get("matches_prepared")),
    }


def _variant_summary(variant: Variant) -> dict[str, Any]:
    profile_exists = variant.launch_profile.exists()
    prepared_exists = variant.prepared.exists()
    audit = _audit_summary(variant.launch_audit)
    ready = (
        prepared_exists
        and profile_exists
        and bool(audit.get("ok"))
        and bool(audit.get("request_ok"))
        and bool(audit.get("safe_to_replay"))
        and bool(audit.get("locked"))
        and bool(audit.get("matches_prepared"))
    )
    return {
        "name": variant.name,
        "prepared": str(variant.prepared),
        "prepared_exists": prepared_exists,
        "prepared_sha256": _sha256_file(variant.prepared),
        "launch_profile": str(variant.launch_profile),
        "launch_profile_exists": profile_exists,
        "launch_profile_sha256": _sha256_file(variant.launch_profile),
        "launch_audit": audit,
        "env": dict(variant.env),
        "ready": ready,
    }


def _result_path(output_dir: Path, prefix: str, variant: Variant, prompt: dict[str, Any], max_new_tokens: int) -> Path:
    label = _safe_label(str(prompt["label"]))
    return output_dir / f"{prefix}-{variant.name}-{label}-max{max_new_tokens}-generate-tokenids.json"


def _argv_for_task(
    *,
    variant: Variant,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    runner: str,
    write_result: Path,
    quiet_runner: bool,
) -> list[str]:
    argv = [
        "python",
        "-m",
        "largerlm",
        "generate-prepared-token-ids",
        str(variant.prepared),
        "--apply-launch-profile",
        str(variant.launch_profile),
        "--lock-launch-profile",
        "--require-locked-launch-profile",
        "--require-launch-audit",
        str(variant.launch_audit),
        "--runner",
        runner,
        "--prompt-token-ids",
        ",".join(str(item) for item in prompt_token_ids),
        "--max-new-tokens",
        str(max_new_tokens),
        "--write-result",
        str(write_result),
    ]
    if quiet_runner:
        argv.append("--quiet-runner")
    return argv


def _shell_join(argv: list[str], env: dict[str, str] | None = None) -> str:
    prefix = []
    for key, value in sorted((env or {}).items()):
        prefix.append(f"{key}={shlex.quote(value)}")
    if prefix:
        return "env " + " ".join(prefix + [shlex.quote(item) for item in argv])
    return " ".join(shlex.quote(item) for item in argv)


def _dedupe_prompts(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for prompt in prompts:
        key = tuple(prompt["prompt_token_ids"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(prompt)
    return deduped


def build_plan(
    *,
    prompt_paths: list[Path],
    variants: list[Variant],
    existing_results: list[tuple[str, Path]] | None = None,
    output_dir: Path,
    result_prefix: str,
    max_new_tokens: int,
    runner: str,
    quiet_runner: bool = True,
    max_prompt_tokens: int | None = None,
    dedupe_prompts: bool = True,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if not variants:
        raise SystemExit("at least one --variant is required")
    prompts = [extract_prompt_record(path) for path in prompt_paths]
    if dedupe_prompts:
        prompts = _dedupe_prompts(prompts)
    for prompt in prompts:
        within = True if max_prompt_tokens is None else prompt["prompt_token_count"] <= max_prompt_tokens
        prompt["within_max_prompt_tokens"] = within
    existing_by_variant_prompt: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for variant_name, result_path in existing_results or []:
        safe_variant_name = _safe_label(variant_name)
        record = extract_prompt_record(result_path)
        key = (safe_variant_name, tuple(record["prompt_token_ids"]))
        existing_by_variant_prompt[key] = {
            "path": str(result_path),
            "sha256": _sha256_file(result_path),
            "prompt_token_count": record["prompt_token_count"],
            "generated_token_ids": record["source_generated_token_ids"],
        }
    output_dir_str = str(output_dir)
    variant_summaries = [_variant_summary(variant) for variant in variants]
    variant_ready = {item["name"]: bool(item["ready"]) for item in variant_summaries}
    tasks: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        for variant in variants:
            planned_result = _result_path(output_dir, result_prefix, variant, prompt, max_new_tokens)
            existing = existing_by_variant_prompt.get(
                (variant.name, tuple(prompt["prompt_token_ids"]))
            )
            effective_result = Path(str(existing["path"])) if existing else planned_result
            argv = _argv_for_task(
                variant=variant,
                prompt_token_ids=prompt["prompt_token_ids"],
                max_new_tokens=max_new_tokens,
                runner=runner,
                write_result=planned_result,
                quiet_runner=quiet_runner,
            )
            prompt_ok = bool(prompt["within_max_prompt_tokens"])
            ready = variant_ready.get(variant.name, False) and prompt_ok
            tasks.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_label": prompt["label"],
                    "variant": variant.name,
                    "prompt_token_count": prompt["prompt_token_count"],
                    "result_path": str(effective_result),
                    "result_exists": effective_result.exists(),
                    "planned_result_path": str(planned_result),
                    "planned_result_exists": planned_result.exists(),
                    "existing_result": existing,
                    "argv": argv,
                    "command": _shell_join(argv, variant.env),
                    "env": dict(variant.env),
                    "ready": ready,
                    "blocked_reasons": [
                        reason
                        for reason, blocked in (
                            ("variant_not_ready", not variant_ready.get(variant.name, False)),
                            ("prompt_exceeds_max_prompt_tokens", not prompt_ok),
                        )
                        if blocked
                    ],
                }
            )
    return {
        "schema": SCHEMA,
        "prompt_results": [str(path) for path in prompt_paths],
        "prompt_count": len(prompts),
        "variant_count": len(variants),
        "task_count": len(tasks),
        "output_dir": output_dir_str,
        "result_prefix": result_prefix,
        "max_new_tokens": max_new_tokens,
        "max_prompt_tokens": max_prompt_tokens,
        "runner": runner,
        "quiet_runner": quiet_runner,
        "dedupe_prompts": dedupe_prompts,
        "existing_result_count": len(existing_results or []),
        "prompts": prompts,
        "variants": variant_summaries,
        "tasks": tasks,
        "all_variants_ready": all(item["ready"] for item in variant_summaries),
        "all_prompts_within_limit": all(prompt["within_max_prompt_tokens"] for prompt in prompts),
        "all_tasks_ready": all(task["ready"] for task in tasks),
    }


def write_shell_script(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "check_no_largerlm_runner() {",
        "  if pgrep -fl 'largerlm|metal/largerlm-runner' >/dev/null; then",
        "    echo 'refusing to start: existing largerlm/Metal runner process found' >&2",
        "    pgrep -fl 'largerlm|metal/largerlm-runner' >&2 || true",
        "    exit 1",
        "  fi",
        "}",
        "",
        f"mkdir -p {shlex.quote(str(plan['output_dir']))}",
        "",
    ]
    for task in _as_list(plan.get("tasks")):
        mapping = _as_mapping(task)
        if not mapping.get("ready"):
            lines.append(f"# skipped not-ready task: {mapping.get('variant')} {mapping.get('prompt_label')}")
            continue
        if mapping.get("result_exists") or mapping.get("planned_result_exists"):
            lines.append(
                f"# skipped existing result: {mapping.get('variant')} "
                f"{mapping.get('prompt_label')} -> {mapping.get('result_path')}"
            )
            continue
        command = mapping.get("command")
        if not isinstance(command, str):
            continue
        lines.extend(
            [
                "check_no_largerlm_runner",
                f"echo 'running {mapping.get('variant')} {mapping.get('prompt_label')}'",
                command,
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _parse_variant(values: list[str]) -> Variant:
    if len(values) != 4:
        raise SystemExit("--variant expects NAME PREPARED PROFILE AUDIT")
    name = _safe_label(values[0])
    if not name:
        raise SystemExit("variant name must not be empty")
    return Variant(
        name=name,
        prepared=Path(values[1]),
        launch_profile=Path(values[2]),
        launch_audit=Path(values[3]),
        env={},
    )


def _apply_variant_env(variants: list[Variant], env_items: list[list[str]] | None) -> list[Variant]:
    env_by_name: dict[str, dict[str, str]] = {variant.name: dict(variant.env) for variant in variants}
    for raw in env_items or []:
        if len(raw) != 2:
            raise SystemExit("--variant-env expects NAME KEY=VALUE")
        name = _safe_label(raw[0])
        if name not in env_by_name:
            raise SystemExit(f"--variant-env references unknown variant {raw[0]!r}")
        key_value = raw[1]
        if "=" not in key_value:
            raise SystemExit("--variant-env value must be KEY=VALUE")
        key, value = key_value.split("=", 1)
        if not key:
            raise SystemExit("--variant-env key must not be empty")
        env_by_name[name][key] = value
    return [
        Variant(
            name=variant.name,
            prepared=variant.prepared,
            launch_profile=variant.launch_profile,
            launch_audit=variant.launch_audit,
            env=env_by_name[variant.name],
        )
        for variant in variants
    ]


def _parse_existing_result(values: list[str]) -> tuple[str, Path]:
    if len(values) != 2:
        raise SystemExit("--existing-result expects VARIANT RESULT_JSON")
    return (_safe_label(values[0]), Path(values[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a safe multi-prompt replay matrix from existing result JSON artifacts."
    )
    parser.add_argument("--prompt-result", action="append", required=True, help="existing result JSON with prompt_token_ids")
    parser.add_argument(
        "--variant",
        action="append",
        nargs=4,
        metavar=("NAME", "PREPARED", "PROFILE", "AUDIT"),
        required=True,
        help="variant replay binding",
    )
    parser.add_argument(
        "--variant-env",
        action="append",
        nargs=2,
        metavar=("NAME", "KEY=VALUE"),
        help="environment variable required for one variant",
    )
    parser.add_argument(
        "--existing-result",
        action="append",
        nargs=2,
        metavar=("VARIANT", "RESULT_JSON"),
        help="reuse an already completed result for a variant/prompt-token match",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-prefix", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--runner", default="metal/largerlm-runner")
    parser.add_argument("--no-quiet-runner", action="store_true")
    parser.add_argument("--no-dedupe-prompts", action="store_true")
    parser.add_argument("--write-plan", default=None)
    parser.add_argument("--write-script", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    variants = [_parse_variant(item) for item in args.variant]
    variants = _apply_variant_env(variants, args.variant_env)
    existing_results = [
        _parse_existing_result(item) for item in (args.existing_result or [])
    ]
    plan = build_plan(
        prompt_paths=[Path(item) for item in args.prompt_result],
        variants=variants,
        existing_results=existing_results,
        output_dir=Path(args.output_dir),
        result_prefix=args.result_prefix,
        max_new_tokens=args.max_new_tokens,
        runner=args.runner,
        quiet_runner=not args.no_quiet_runner,
        max_prompt_tokens=args.max_prompt_tokens,
        dedupe_prompts=not args.no_dedupe_prompts,
    )
    if args.write_plan:
        path = Path(args.write_plan)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_script:
        path = Path(args.write_script)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_shell_script(path, plan)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print(
            f"multi-prompt replay plan: prompts={plan['prompt_count']} "
            f"variants={plan['variant_count']} tasks={plan['task_count']} "
            f"ready={plan['all_tasks_ready']}"
        )
        for task in plan["tasks"]:
            status = "ready" if task["ready"] else "blocked"
            reasons = ",".join(task["blocked_reasons"])
            suffix = f" ({reasons})" if reasons else ""
            existing = " existing" if task.get("existing_result") else ""
            print(
                f"  {status}{existing}: {task['variant']} {task['prompt_label']} "
                f"tokens={task['prompt_token_count']} -> {task['result_path']}{suffix}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
