from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .prefill_backend import _file_sha256, default_prefill_backend_probe_path
from .safety import DiskBudget, SafetyError, disk_budget
from .safetensors import (
    HEADER_MANIFEST_NAME,
    SafetensorsError,
    validate_local_safetensors_header,
)


class ArtifactStatusError(RuntimeError):
    """Raised when a checkpoint artifact status cannot be inspected."""


GLM_5_2_BRINGUP_MAX_CACHE_GIB = "16"
GLM_5_2_BRINGUP_PREPARE_DISK_MARGIN_GIB = "32"
GLM_5_2_BRINGUP_UNIFIED_MEMORY_GIB = "128"
GLM_5_2_BRINGUP_GROUP_SIZE = "32"
GLM_5_2_BRINGUP_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS = "30"
GLM_5_2_BRINGUP_LAUNCH_AUDIT_PROMPT_TOKENS = "2048"
GLM_5_2_BRINGUP_MAX_NEW_TOKENS = "1"
GLM_5_2_BRINGUP_PREFILL_ACCELERATION_FLAGS: tuple[str, ...] = ()
GLM_5_2_BRINGUP_NON_ACCELERATED_PREFILL_AUDIT_FLAGS = (
    "--allow-non-accelerated-prefill-launch-audit",
)
GLM_5_2_BRINGUP_PREFILL_GUARD_FLAGS = (
    "--prefill-prompt-chunk-tokens",
    "17",
    "--prefill-max-routed-read-amplification",
    "67.2",
    "--prefill-max-routed-read-gib",
    "24097.5",
    "--prefill-max-routed-read-seconds",
    "4074.3",
    "--prefill-max-stage-mib",
    "2731.61",
    "--prefill-max-compact-stage-mib",
    "2731.05",
    "--prefill-max-stage-raw-ranges",
    "143",
    "--prefill-max-stage-coalesced-ranges",
    "143",
    "--prefill-expert-stage-tiling",
)
GLM_5_2_BRINGUP_PREFILL_GENERATION_ONLY_FLAGS = (
    "--prefill-static-capacity-per-expert",
    "auto",
)
GLM_5_2_BRINGUP_CHECK_METAL_FINAL_LOGITS_FLAGS = (
    "--check-metal-final-logits",
)
GLM_5_2_BRINGUP_METAL_FINAL_LOGITS_FLAGS = (
    "--metal-final-logits",
)
DOWNSTREAM_ARTIFACT_JSON_MAX_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class ShardFileStatus:
    name: str
    present: bool
    expected_file_bytes: int | None
    actual_file_bytes: int | None
    complete: bool | None
    issue: str | None
    url: str | None
    header_checked: bool = False
    header_ok: bool | None = None
    header_error: str | None = None
    header_data_start: int | None = None
    header_tensor_count: int | None = None


@dataclass(frozen=True)
class BringupPlanStep:
    step_id: str
    title: str
    command: tuple[str, ...] | None
    step_status: str
    command_available: bool
    blocked_reason: str | None
    prerequisite_step_ids: tuple[str, ...]
    reads_weight_payloads: bool
    writes_artifacts: bool
    runs_model: bool


@dataclass(frozen=True)
class CheckpointArtifactStatus:
    model_dir: Path
    config_present: bool
    index_present: bool
    header_manifest_present: bool
    header_manifest_valid: bool
    header_manifest_error: str | None
    source_repo: str | None
    source_revision: str | None
    source_endpoint: str | None
    expected_tensor_bytes: int | None
    expected_safetensors_file_bytes: int | None
    remaining_safetensors_file_bytes: int | None
    present_safetensors_file_bytes: int
    download_disk_budget: DiskBudget | None
    download_disk_ok: bool | None
    local_header_check_requested: bool
    local_header_checked_shard_count: int
    local_header_ok_shard_count: int
    local_header_error_count: int
    local_headers_ok: bool | None
    artifact_clean: bool
    expected_shard_count: int
    present_shard_count: int
    complete_shard_count: int
    missing_shard_count: int
    partial_shard_count: int
    extra_shard_count: int
    download_complete: bool
    download_complete_proven: bool
    can_run_metadata_preflight: bool
    can_attempt_prepare: bool
    hf_download_command: tuple[str, ...] | None
    header_fetch_command: tuple[str, ...] | None
    download_precheck_command: tuple[str, ...] | None
    post_copy_check_command: tuple[str, ...] | None
    prefill_backend_command: tuple[str, ...] | None
    preflight_command: tuple[str, ...] | None
    prepare_dry_run_command: tuple[str, ...] | None
    prepare_execute_command: tuple[str, ...] | None
    inspect_prepared_command: tuple[str, ...] | None
    launch_audit_command: tuple[str, ...] | None
    minimal_smoke_command: tuple[str, ...] | None
    prepared_manifest_present: bool
    prepared_manifest_valid: bool
    prepared_manifest_error: str | None
    launch_profile_present: bool
    launch_profile_valid: bool
    launch_profile_error: str | None
    launch_audit_present: bool
    launch_audit_valid: bool
    launch_audit_error: str | None
    minimal_smoke_result_present: bool
    minimal_smoke_result_valid: bool
    minimal_smoke_result_error: str | None
    prefill_backend_report_present: bool
    prefill_backend_report_valid: bool
    prefill_backend_report_error: str | None
    preflight_report_present: bool
    preflight_report_valid: bool
    preflight_report_error: str | None
    prepare_dry_run_report_present: bool
    prepare_dry_run_report_valid: bool
    prepare_dry_run_report_error: str | None
    bringup_plan: tuple[BringupPlanStep, ...]
    next_bringup_step: BringupPlanStep | None
    issues: tuple[str, ...]
    shards: tuple[ShardFileStatus, ...]
    extra_shards: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.can_run_metadata_preflight and not self.issues


@dataclass(frozen=True)
class MissingShardDownloadEntry:
    name: str
    url: str | None
    issue: str | None
    expected_file_bytes: int | None
    actual_file_bytes: int | None
    remaining_file_bytes: int | None


def missing_shard_download_entries(
    status: CheckpointArtifactStatus,
) -> tuple[MissingShardDownloadEntry, ...]:
    entries: list[MissingShardDownloadEntry] = []
    for shard in status.shards:
        if shard.complete is True:
            continue
        remaining = None
        if shard.expected_file_bytes is not None:
            remaining = max(
                shard.expected_file_bytes - int(shard.actual_file_bytes or 0),
                0,
            )
        entries.append(
            MissingShardDownloadEntry(
                name=shard.name,
                url=shard.url,
                issue=shard.issue,
                expected_file_bytes=shard.expected_file_bytes,
                actual_file_bytes=shard.actual_file_bytes,
                remaining_file_bytes=remaining,
            )
        )
    return tuple(entries)


def missing_shard_download_manifest(
    status: CheckpointArtifactStatus,
) -> dict[str, object]:
    entries = missing_shard_download_entries(status)
    return {
        "version": 1,
        "model_dir": str(status.model_dir),
        "source": {
            "repo": status.source_repo,
            "revision": status.source_revision,
            "endpoint": status.source_endpoint,
        },
        "download_complete": status.download_complete,
        "download_complete_proven": status.download_complete_proven,
        "expected_safetensors_file_bytes": status.expected_safetensors_file_bytes,
        "present_safetensors_file_bytes": status.present_safetensors_file_bytes,
        "remaining_safetensors_file_bytes": status.remaining_safetensors_file_bytes,
        "download_disk_ok": status.download_disk_ok,
        "download_disk_budget": (
            None
            if status.download_disk_budget is None
            else {
                "output_dir": str(status.download_disk_budget.output_dir),
                "required_bytes": status.download_disk_budget.required_bytes,
                "available_bytes": status.download_disk_budget.available_bytes,
                "safety_margin_bytes": status.download_disk_budget.safety_margin_bytes,
                "ok": status.download_disk_budget.ok,
            }
        ),
        "entries": [
            {
                "name": entry.name,
                "url": entry.url,
                "issue": entry.issue,
                "expected_file_bytes": entry.expected_file_bytes,
                "actual_file_bytes": entry.actual_file_bytes,
                "remaining_file_bytes": entry.remaining_file_bytes,
            }
            for entry in entries
        ],
    }


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise ArtifactStatusError(f"failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactStatusError(f"failed to parse {path}: {exc}") from exc


def _load_optional_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, f"failed to stat {path}: {exc}"
    if size > DOWNSTREAM_ARTIFACT_JSON_MAX_BYTES:
        return (
            None,
            f"{path} is larger than the {DOWNSTREAM_ARTIFACT_JSON_MAX_BYTES} byte "
            "status safety cap",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"failed to read {path}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"failed to parse {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{path} must contain a JSON object"
    return payload, None


def _relative_json_path(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _prepared_manifest_json_check(path: Path) -> tuple[bool, str | None]:
    payload, error = _load_optional_json_object(path)
    if error is not None:
        return False, error
    if payload is None:
        return False, None
    if payload.get("version") != 1:
        return False, f"{path} version must be 1"
    model_dir = payload.get("model_dir")
    if not isinstance(model_dir, str) or not model_dir:
        return False, f"{path} model_dir must be a non-empty string"
    for field in (
        "experts_layout",
        "resident_layout",
        "decode_cache_layout",
        "decode_cache_file",
    ):
        if not _relative_json_path(payload.get(field)):
            return False, f"{path} {field} must be a relative path"
    return True, None


def _extract_launch_profile_payload(
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    if isinstance(payload.get("argv"), list):
        return payload
    for key in (
        "request_launch_profile",
        "suggested_launch_profile",
        "combined_launch_profile",
        "profile",
    ):
        value = payload.get(key)
        if isinstance(value, dict) and isinstance(value.get("argv"), list):
            return value
    return None


def _launch_profile_json_check(path: Path) -> tuple[bool, str | None]:
    payload, error = _load_optional_json_object(path)
    if error is not None:
        return False, error
    if payload is None:
        return False, None
    profile = _extract_launch_profile_payload(payload)
    if profile is None:
        return (
            False,
            f"{path} must contain argv, request_launch_profile, "
            "suggested_launch_profile, combined_launch_profile, or profile",
        )
    argv = profile.get("argv")
    if not isinstance(argv, list) or not argv:
        return False, f"{path} launch profile argv must be a non-empty array"
    if any(not isinstance(item, str) or not item for item in argv):
        return False, f"{path} launch profile argv entries must be strings"
    if profile.get("argv_safe_to_replay") is False:
        return False, f"{path} launch profile argv is marked unsafe to replay"
    return True, None


def _launch_audit_json_check(path: Path) -> tuple[bool, str | None]:
    payload, error = _load_optional_json_object(path)
    if error is not None:
        return False, error
    if payload is None:
        return False, None
    if payload.get("schema") != "largerlm.launch_audit.v1":
        return False, f"{path} schema must be largerlm.launch_audit.v1"
    audit = payload.get("launch_audit")
    if not isinstance(audit, dict):
        return False, f"{path} launch_audit must be an object"
    if audit.get("ok") is not True:
        return False, f"{path} launch_audit.ok must be true"
    checks = audit.get("checks")
    if not isinstance(checks, list) or not checks:
        return False, f"{path} launch_audit.checks must be a non-empty array"
    if any(not isinstance(check, dict) for check in checks):
        return False, f"{path} launch_audit.checks entries must be objects"
    return True, None


def _preflight_report_json_check(
    path: Path,
    *,
    model_dir: Path,
) -> tuple[bool, str | None]:
    payload, error = _load_optional_json_object(path)
    if error is not None:
        return False, error
    if payload is None:
        return False, None
    if payload.get("ok") is not True:
        return False, f"{path} ok must be true"
    if payload.get("model_dir") != str(model_dir):
        return False, f"{path} model_dir must be {model_dir}"
    public_shape = payload.get("public_glm_5_2_shape")
    if not isinstance(public_shape, dict):
        return False, f"{path} public_glm_5_2_shape must be an object"
    if public_shape.get("matches") is not True:
        return False, f"{path} public_glm_5_2_shape.matches must be true"
    return True, None


def _prepare_dry_run_report_json_check(
    path: Path,
    *,
    model_dir: Path,
    output_dir: Path,
) -> tuple[bool, str | None]:
    payload, error = _load_optional_json_object(path)
    if error is not None:
        return False, error
    if payload is None:
        return False, None
    if payload.get("ok") is not True:
        return False, f"{path} ok must be true"
    if payload.get("executed") is not False:
        return False, f"{path} executed must be false"
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        return False, f"{path} paths must be an object"
    if paths.get("output_dir") != str(output_dir):
        return False, f"{path} paths.output_dir must be {output_dir}"
    preflight = payload.get("preflight")
    if not isinstance(preflight, dict):
        return False, f"{path} preflight must be an object"
    if preflight.get("model_dir") != str(model_dir):
        return False, f"{path} preflight.model_dir must be {model_dir}"
    if preflight.get("ok") is not True:
        return False, f"{path} preflight.ok must be true"
    public_shape = preflight.get("public_glm_5_2_shape")
    if not isinstance(public_shape, dict):
        return False, f"{path} preflight.public_glm_5_2_shape must be an object"
    if public_shape.get("matches") is not True:
        return False, f"{path} preflight.public_glm_5_2_shape.matches must be true"
    return True, None


def _prefill_backend_report_json_check(path: Path) -> tuple[bool, str | None]:
    payload, error = _load_optional_json_object(path)
    if error is not None:
        return False, error
    if payload is None:
        return False, None
    if payload.get("host_probe_requested") is not True:
        return False, f"{path} host_probe_requested must be true"
    if payload.get("host_probe_ran") is not True:
        return False, f"{path} host_probe_ran must be true"
    if payload.get("host_probe_ok") is not True:
        return False, f"{path} host_probe_ok must be true"
    expected_probe = str(default_prefill_backend_probe_path())
    if payload.get("host_probe_path") != expected_probe:
        return False, f"{path} host_probe_path must be {expected_probe}"
    reported_probe_hash = payload.get("host_probe_sha256")
    if reported_probe_hash is not None:
        if not isinstance(reported_probe_hash, str) or not reported_probe_hash:
            return False, f"{path} host_probe_sha256 must be a non-empty string"
        current_probe_hash = _file_sha256(default_prefill_backend_probe_path())
        if current_probe_hash != reported_probe_hash:
            return False, f"{path} host_probe_sha256 does not match current probe"
    if payload.get("mps_graph_probe_requested") is not True:
        return False, f"{path} mps_graph_probe_requested must be true"
    if payload.get("mps_graph_probe_ran") is not True:
        return False, f"{path} mps_graph_probe_ran must be true"
    if payload.get("mps_graph_probe_ok") is not True:
        return False, f"{path} mps_graph_probe_ok must be true"
    if payload.get("mps_graph_runtime_available") is not True:
        return False, f"{path} mps_graph_runtime_available must be true"
    if payload.get("mpp_compile_probe_requested") is not True:
        return False, f"{path} mpp_compile_probe_requested must be true"
    if payload.get("mpp_compile_probe_ran") is not True:
        return False, f"{path} mpp_compile_probe_ran must be true"
    if payload.get("mpp_run_probe_requested") is not True:
        return False, f"{path} mpp_run_probe_requested must be true"
    if payload.get("mpp_run_probe_ran") is not True:
        return False, f"{path} mpp_run_probe_ran must be true"
    probe_timeout = payload.get("probe_timeout_seconds")
    try:
        probe_timeout_value = float(probe_timeout)
    except (TypeError, ValueError):
        return False, f"{path} probe_timeout_seconds must be numeric"
    if probe_timeout_value < float(GLM_5_2_BRINGUP_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS):
        return (
            False,
            f"{path} probe_timeout_seconds must be at least "
            f"{GLM_5_2_BRINGUP_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS}",
        )
    validated = payload.get("validated_accelerated_prefill_backends")
    if not isinstance(validated, list) or "mpsgraph-f32" not in validated:
        return (
            False,
            f"{path} validated_accelerated_prefill_backends must include mpsgraph-f32",
        )
    suggested = payload.get("suggested_prefill_acceleration_flags")
    if not isinstance(suggested, dict):
        return False, f"{path} suggested_prefill_acceleration_flags must be an object"
    if suggested.get("runtime_probe_satisfied") is not True:
        return False, f"{path} runtime_probe_satisfied must be true"
    neural = payload.get("prefill_neural_accelerator_status")
    if not isinstance(neural, dict):
        return False, f"{path} prefill_neural_accelerator_status must be an object"
    return True, None


def _json_int_list(value: object) -> list[int] | None:
    if not isinstance(value, list):
        return None
    items: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        items.append(int(item))
    return items


def _minimal_smoke_result_json_check(
    path: Path,
    *,
    prepared_manifest_path: Path,
    launch_profile_path: Path,
    launch_audit_path: Path,
) -> tuple[bool, str | None]:
    payload, error = _load_optional_json_object(path)
    if error is not None:
        return False, error
    if payload is None:
        return False, None
    if payload.get("schema") != "largerlm.prepared_token_generation_result.v1":
        return (
            False,
            f"{path} schema must be largerlm.prepared_token_generation_result.v1",
        )
    if payload.get("source") != "generate_prepared_token_ids":
        return False, f"{path} source must be generate_prepared_token_ids"
    expected_manifest = str(prepared_manifest_path)
    if payload.get("prepared_manifest") != expected_manifest:
        return (
            False,
            f"{path} prepared_manifest must be {expected_manifest}",
        )
    request = payload.get("request")
    if not isinstance(request, dict):
        return False, f"{path} request must be an object"
    request_prompt = _json_int_list(request.get("prompt_token_ids"))
    if request_prompt != [0]:
        return False, f"{path} request.prompt_token_ids must be [0]"
    if request.get("max_new_tokens") != 1:
        return False, f"{path} request.max_new_tokens must be 1"
    expected_audit = str(launch_audit_path)
    if request.get("launch_audit_path") != expected_audit:
        return False, f"{path} request.launch_audit_path must be {expected_audit}"
    token_result = payload.get("token_result")
    if not isinstance(token_result, dict):
        return False, f"{path} token_result must be an object"
    result_prompt = _json_int_list(token_result.get("prompt_token_ids"))
    if result_prompt != [0]:
        return False, f"{path} token_result.prompt_token_ids must be [0]"
    generated = _json_int_list(token_result.get("generated_token_ids"))
    if not generated:
        return False, f"{path} token_result.generated_token_ids must be non-empty"
    steps = token_result.get("steps")
    if not isinstance(steps, list):
        return False, f"{path} token_result.steps must be an array"
    applied = token_result.get("applied_launch_profile")
    if not isinstance(applied, dict):
        return False, f"{path} token_result.applied_launch_profile must be an object"
    expected_profile = str(launch_profile_path)
    if applied.get("path") != expected_profile:
        return (
            False,
            f"{path} token_result.applied_launch_profile.path must be "
            f"{expected_profile}",
        )
    if applied.get("locked") is not True:
        return False, f"{path} token_result.applied_launch_profile.locked must be true"
    if applied.get("lock_required") is not True:
        return (
            False,
            f"{path} token_result.applied_launch_profile.lock_required must be true"
        )
    if applied.get("matches_prepared") is not True:
        return (
            False,
            f"{path} token_result.applied_launch_profile.matches_prepared must be true"
        )
    if applied.get("current_prepared_manifest") != expected_manifest:
        return (
            False,
            f"{path} token_result.applied_launch_profile.current_prepared_manifest "
            f"must be {expected_manifest}",
        )
    return True, None


def _non_negative_int(value: object, *, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SafetensorsError(f"{path} {field} must be a non-negative integer")
    return value


def _index_total_size(index: dict[str, Any], *, path: Path) -> int | None:
    metadata = index.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise SafetensorsError(f"{path} metadata must be an object")
    total_size = metadata.get("total_size")
    if total_size is None:
        return None
    return _non_negative_int(total_size, field="metadata.total_size", path=path)


def _weight_map_shards(index: dict[str, Any], *, path: Path) -> tuple[str, ...]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise SafetensorsError(f"{path} does not contain a weight_map")
    shards: set[str] = set()
    for tensor_name, shard in weight_map.items():
        if not isinstance(shard, str) or not shard:
            raise SafetensorsError(
                f"{path} weight_map shard for tensor {tensor_name!r} must be "
                "a non-empty string"
            )
        shard_path = Path(shard)
        if shard_path.is_absolute() or any(part == ".." for part in shard_path.parts):
            raise SafetensorsError(
                f"{path} weight_map shard for tensor {tensor_name!r} escapes "
                f"model directory: {shard!r}"
            )
        shards.add(shard)
    return tuple(sorted(shards))


def _source_field(source: object, field: str) -> str | None:
    if not isinstance(source, dict):
        return None
    value = source.get(field)
    return value if isinstance(value, str) and value else None


def _resolve_url(endpoint: str, repo: str, revision: str, path: str) -> str:
    from urllib.parse import quote

    endpoint = endpoint.rstrip("/")
    repo_q = quote(repo.strip("/"), safe="/")
    revision_q = quote(revision, safe="")
    path_q = quote(path, safe="/")
    return f"{endpoint}/{repo_q}/resolve/{revision_q}/{path_q}"


def _format_gib_arg(byte_count: int) -> str:
    return f"{byte_count / 1024**3:.12g}"


def _checkpoint_status_command(
    root: Path,
    *,
    source_repo: str | None,
    revision_value: str,
    endpoint_value: str,
    download_disk_safety_margin_bytes: int | None = None,
    extra_flags: tuple[str, ...] = (),
) -> tuple[str, ...]:
    command: tuple[str, ...] = (
        "python",
        "-m",
        "largerlm",
        "checkpoint-status",
        str(root),
    )
    if source_repo is not None:
        command += (
            "--repo",
            source_repo,
            "--revision",
            revision_value,
            "--endpoint",
            endpoint_value,
        )
    if download_disk_safety_margin_bytes is not None:
        command += (
            "--download-disk-margin-gib",
            _format_gib_arg(download_disk_safety_margin_bytes),
        )
    command += extra_flags
    command += ("--json",)
    return command


def _step(
    step_id: str,
    title: str,
    command: tuple[str, ...] | None,
    *,
    blocked_reason: str | None,
    completed: bool = False,
    prerequisite_step_ids: tuple[str, ...] = (),
    reads_weight_payloads: bool = False,
    writes_artifacts: bool = False,
    runs_model: bool = False,
) -> BringupPlanStep:
    if completed:
        step_status = "complete"
        command_available = False
        blocked_reason = None
    elif blocked_reason is not None:
        step_status = "blocked"
        command_available = False
    elif command is not None:
        step_status = "ready"
        command_available = True
    else:
        step_status = "blocked"
        command_available = False
        blocked_reason = "command is unavailable"
    return BringupPlanStep(
        step_id=step_id,
        title=title,
        command=command,
        step_status=step_status,
        command_available=command_available,
        blocked_reason=blocked_reason,
        prerequisite_step_ids=prerequisite_step_ids,
        reads_weight_payloads=reads_weight_payloads,
        writes_artifacts=writes_artifacts,
        runs_model=runs_model,
    )


def _prepare_execute_blocked_reason(
    *,
    can_attempt_prepare: bool,
    artifact_clean: bool,
    header_manifest_valid: bool,
    local_headers_ok: bool | None,
) -> str | None:
    if not can_attempt_prepare:
        return "complete local safetensors shards are not ready"
    if not artifact_clean:
        return "checkpoint artifact is not clean"
    if header_manifest_valid and local_headers_ok is not True:
        return "run checkpoint-status --verify-local-headers before execute"
    return None


def inspect_checkpoint_artifact(
    model_dir: str | Path,
    *,
    repo: str | None = None,
    revision: str | None = None,
    endpoint: str | None = None,
    download_disk_safety_margin_bytes: int = 16 * 1024**3,
    verify_local_headers: bool = False,
) -> CheckpointArtifactStatus:
    root = Path(model_dir)
    config_present = (root / "config.json").exists()
    index_path = root / "model.safetensors.index.json"
    manifest_path = root / HEADER_MANIFEST_NAME
    index_present = index_path.exists()
    header_manifest_present = manifest_path.exists()
    header_manifest_valid = False
    header_manifest_error: str | None = None
    manifest_index: dict[str, Any] | None = None
    manifest_shards: dict[str, Any] = {}
    source_repo: str | None = repo
    source_revision: str | None = revision
    source_endpoint: str | None = endpoint
    issues: list[str] = []

    if header_manifest_present:
        try:
            manifest = _load_json(manifest_path)
            if not isinstance(manifest, dict):
                raise SafetensorsError(f"{manifest_path} must contain a JSON object")
            if manifest.get("version") != 1:
                raise SafetensorsError(f"{manifest_path} version must be 1")
            index_obj = manifest.get("index")
            shards_obj = manifest.get("shards")
            if not isinstance(index_obj, dict):
                raise SafetensorsError(f"{manifest_path} must contain an index object")
            if not isinstance(shards_obj, dict):
                raise SafetensorsError(f"{manifest_path} must contain a shards object")
            manifest_index = index_obj
            manifest_shards = shards_obj
            source = manifest.get("source")
            source_repo = source_repo or _source_field(source, "repo")
            source_revision = source_revision or _source_field(source, "revision")
            source_endpoint = source_endpoint or _source_field(source, "endpoint")
            header_manifest_valid = True
        except (ArtifactStatusError, SafetensorsError) as exc:
            header_manifest_error = str(exc)
            issues.append(f"header manifest is invalid: {exc}")

    index: dict[str, Any] | None = None
    if manifest_index is not None:
        index = manifest_index
    elif index_present:
        loaded = _load_json(index_path)
        if not isinstance(loaded, dict):
            raise ArtifactStatusError(f"{index_path} must contain a JSON object")
        index = loaded
    else:
        issues.append("model.safetensors.index.json is missing")

    expected_tensor_bytes: int | None = None
    expected_shards: tuple[str, ...] = ()
    if index is not None:
        source_path = manifest_path if manifest_index is not None else index_path
        try:
            expected_tensor_bytes = _index_total_size(index, path=source_path)
            expected_shards = _weight_map_shards(index, path=source_path)
        except SafetensorsError as exc:
            issues.append(str(exc))

    local_shards = {path.name: path for path in root.glob("*.safetensors")}
    expected_set = set(expected_shards)
    extra_names = tuple(sorted(set(local_shards) - expected_set)) if expected_set else ()
    endpoint_value = source_endpoint or "https://huggingface.co"
    revision_value = source_revision or "main"
    expected_file_bytes_total = 0
    expected_file_bytes_known = True
    present_file_bytes = 0
    remaining_file_bytes = 0
    complete_count = 0
    missing_count = 0
    partial_count = 0
    local_header_checked_count = 0
    local_header_ok_count = 0
    local_header_error_count = 0
    shard_statuses: list[ShardFileStatus] = []
    for shard in expected_shards:
        expected_bytes: int | None = None
        expected_data_start: int | None = None
        expected_header: dict[str, Any] | None = None
        if header_manifest_valid:
            info = manifest_shards.get(shard)
            if isinstance(info, dict):
                try:
                    expected_bytes = _non_negative_int(
                        info.get("file_size"),
                        field=f"shards[{shard!r}].file_size",
                        path=manifest_path,
                    )
                    expected_data_start = _non_negative_int(
                        info.get("data_start"),
                        field=f"shards[{shard!r}].data_start",
                        path=manifest_path,
                    )
                    if expected_bytes < expected_data_start:
                        raise SafetensorsError(
                            f"{manifest_path} shards[{shard!r}].file_size must "
                            "be greater than or equal to data_start"
                        )
                    header_obj = info.get("header")
                    if verify_local_headers and not isinstance(header_obj, dict):
                        raise SafetensorsError(
                            f"{manifest_path} shards[{shard!r}].header must be "
                            "an object"
                        )
                    expected_header = header_obj if isinstance(header_obj, dict) else None
                except SafetensorsError as exc:
                    issues.append(str(exc))
                    expected_file_bytes_known = False
            else:
                issues.append(f"header manifest is missing shard {shard!r}")
                expected_file_bytes_known = False
        else:
            expected_file_bytes_known = False
        if expected_bytes is not None:
            expected_file_bytes_total += expected_bytes
        path = local_shards.get(shard)
        actual_bytes: int | None = None
        present = path is not None
        if present and path is not None:
            try:
                actual_bytes = path.stat().st_size
            except OSError as exc:
                issues.append(f"failed to stat shard {path}: {exc}")
            if actual_bytes is not None:
                present_file_bytes += actual_bytes
        if expected_bytes is not None:
            remaining_file_bytes += max(expected_bytes - int(actual_bytes or 0), 0)
        issue: str | None = None
        complete: bool | None = None
        if not present:
            missing_count += 1
            issue = "missing"
            complete = False
        elif expected_bytes is None:
            complete = None
        elif actual_bytes == expected_bytes:
            complete = True
            complete_count += 1
        else:
            partial_count += 1
            complete = False
            if actual_bytes is None:
                issue = "stat_failed"
            elif actual_bytes < expected_bytes:
                issue = "truncated"
            else:
                issue = "size_mismatch"
        url = None
        if source_repo is not None:
            url = _resolve_url(endpoint_value, source_repo, revision_value, shard)
        header_checked = False
        header_ok: bool | None = None
        header_error: str | None = None
        header_data_start: int | None = None
        header_tensor_count: int | None = None
        if verify_local_headers and present:
            header_checked = True
            local_header_checked_count += 1
            check = validate_local_safetensors_header(
                root,
                shard,
                index=index,
                expected_data_start=expected_data_start,
                expected_file_size=expected_bytes,
                expected_header=expected_header,
            )
            header_ok = check.ok
            header_error = check.error
            header_data_start = check.data_start
            header_tensor_count = check.tensor_count
            if check.ok:
                local_header_ok_count += 1
            else:
                local_header_error_count += 1
                issues.append(
                    f"local safetensors header check failed for {shard}: "
                    f"{check.error}"
                )
        shard_statuses.append(
            ShardFileStatus(
                name=shard,
                present=present,
                expected_file_bytes=expected_bytes,
                actual_file_bytes=actual_bytes,
                complete=complete,
                issue=issue,
                url=url,
                header_checked=header_checked,
                header_ok=header_ok,
                header_error=header_error,
                header_data_start=header_data_start,
                header_tensor_count=header_tensor_count,
            )
        )

    for extra in extra_names:
        path = local_shards[extra]
        try:
            present_file_bytes += path.stat().st_size
        except OSError as exc:
            issues.append(f"failed to stat extra shard {path}: {exc}")

    expected_file_bytes = (
        expected_file_bytes_total if expected_shards and expected_file_bytes_known else None
    )
    remaining_safetensors_file_bytes = (
        remaining_file_bytes if expected_file_bytes is not None else None
    )
    download_disk_budget = None
    download_disk_ok: bool | None = None
    if remaining_safetensors_file_bytes is not None:
        try:
            download_disk_budget = disk_budget(
                root,
                remaining_safetensors_file_bytes,
                safety_margin_bytes=download_disk_safety_margin_bytes,
            )
            download_disk_ok = download_disk_budget.ok
            if not download_disk_budget.ok:
                issues.append(
                    "free disk is below remaining safetensors download bytes "
                    "plus safety margin"
                )
        except SafetyError as exc:
            issues.append(f"download disk budget is invalid: {exc}")
    present_shards = sum(1 for status in shard_statuses if status.present)
    download_complete_proven = bool(
        expected_shards
        and header_manifest_valid
        and missing_count == 0
        and partial_count == 0
        and complete_count == len(expected_shards)
    )
    download_complete = bool(
        expected_shards
        and missing_count == 0
        and partial_count == 0
        and (download_complete_proven or not expected_file_bytes_known)
    )
    local_headers_ok = None
    if verify_local_headers and local_header_checked_count > 0:
        local_headers_ok = local_header_error_count == 0
    can_run_metadata_preflight = bool(
        config_present
        and expected_shards
        and (
            header_manifest_valid
            or (index_present and missing_count == 0 and partial_count == 0)
        )
    )
    can_attempt_prepare = bool(
        config_present and download_complete and local_headers_ok is not False
    )

    if not config_present:
        issues.append("config.json is missing")
    if extra_names:
        issues.append(f"unexpected safetensors shards present: {', '.join(extra_names[:4])}")
    artifact_clean = bool(
        not issues
        and missing_count == 0
        and partial_count == 0
        and not extra_names
        and local_headers_ok is not False
    )

    hf_download_command = None
    header_fetch_command = None
    download_precheck_command = None
    post_copy_check_command = None
    prefill_backend_command = None
    if source_repo is not None:
        hf_download_command = (
            "hf",
            "download",
            source_repo,
            "--revision",
            revision_value,
            "--local-dir",
            str(root),
        )
        header_fetch_command = (
            "python",
            "scripts/fetch_safetensors_headers.py",
            source_repo,
            "--revision",
            revision_value,
            "--endpoint",
            endpoint_value,
            "--fetch-small-files",
            "--output-dir",
            str(root),
        )
    if config_present or index is not None or header_manifest_present:
        download_precheck_command = _checkpoint_status_command(
            root,
            source_repo=source_repo,
            revision_value=revision_value,
            endpoint_value=endpoint_value,
            download_disk_safety_margin_bytes=download_disk_safety_margin_bytes,
            extra_flags=("--require-download-disk-ok",),
        )
        post_copy_check_command = _checkpoint_status_command(
            root,
            source_repo=source_repo,
            revision_value=revision_value,
            endpoint_value=endpoint_value,
            extra_flags=(
                "--verify-local-headers",
                "--require-complete",
                "--require-clean",
            ),
        )
    preflight_command = None
    prepare_dry_run_command = None
    prepare_execute_command = None
    inspect_prepared_command = None
    launch_audit_command = None
    minimal_smoke_command = None
    prepared_dir = root / "largerlm-prepared"
    prefill_backend_report_path = prepared_dir / "prefill-backend-report.json"
    preflight_report_path = prepared_dir / "preflight-report.json"
    prepare_dry_run_report_path = prepared_dir / "prepare-dry-run-report.json"
    launch_profile_path = prepared_dir / "launch-profile.json"
    launch_audit_path = prepared_dir / "launch-audit.json"
    minimal_smoke_result_path = prepared_dir / "minimal-smoke.json"
    prefill_backend_command = (
        "python",
        "-m",
        "largerlm",
        "prefill-backend",
        "--run-mpsgraph-probe",
        "--run-mpp-probe",
        "--probe-timeout-seconds",
        GLM_5_2_BRINGUP_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
        "--write-report",
        str(prefill_backend_report_path),
        "--json",
    )
    if config_present:
        metadata_only_argv = (
            ("--metadata-only",)
            if header_manifest_valid and not download_complete_proven
            else ()
        )
        preflight_command = (
            "python",
            "-m",
            "largerlm",
            "preflight-glm",
            str(root),
            "--quant-bits",
            "4",
            "--group-size",
            GLM_5_2_BRINGUP_GROUP_SIZE,
            "--require-public-glm-5-2-shape",
            "--max-cache-gib",
            GLM_5_2_BRINGUP_MAX_CACHE_GIB,
            "--disk-margin-gib",
            GLM_5_2_BRINGUP_PREPARE_DISK_MARGIN_GIB,
            "--unified-memory-gib",
            GLM_5_2_BRINGUP_UNIFIED_MEMORY_GIB,
            *metadata_only_argv,
            "--write-report",
            str(preflight_report_path),
            "--json",
        )
        prepare_dry_run_command = (
            "python",
            "-m",
            "largerlm",
            "prepare-glm",
            str(root),
            "--output-dir",
            str(prepared_dir),
            "--auto-context-from-budget",
            "--quant-bits",
            "4",
            "--group-size",
            GLM_5_2_BRINGUP_GROUP_SIZE,
            "--require-public-glm-5-2-shape",
            "--max-cache-gib",
            GLM_5_2_BRINGUP_MAX_CACHE_GIB,
            "--disk-margin-gib",
            GLM_5_2_BRINGUP_PREPARE_DISK_MARGIN_GIB,
            "--unified-memory-gib",
            GLM_5_2_BRINGUP_UNIFIED_MEMORY_GIB,
            *metadata_only_argv,
            "--write-report",
            str(prepare_dry_run_report_path),
            "--json",
        )
        if (
            can_attempt_prepare
            and artifact_clean
            and (not header_manifest_valid or local_headers_ok is True)
        ):
            prepare_execute_command = (
                "python",
                "-m",
                "largerlm",
                "prepare-glm",
                str(root),
                "--output-dir",
                str(prepared_dir),
                "--auto-context-from-budget",
                "--quant-bits",
                "4",
                "--group-size",
                GLM_5_2_BRINGUP_GROUP_SIZE,
                "--require-public-glm-5-2-shape",
                "--max-cache-gib",
                GLM_5_2_BRINGUP_MAX_CACHE_GIB,
                "--disk-margin-gib",
                GLM_5_2_BRINGUP_PREPARE_DISK_MARGIN_GIB,
                "--unified-memory-gib",
                GLM_5_2_BRINGUP_UNIFIED_MEMORY_GIB,
                "--auto-cold-read-benchmark",
                "--cold-read-benchmark-mib",
                "1024",
                "--cold-read-benchmark-chunk-mib",
                "8",
                "--execute",
                "--json",
            )
            inspect_prepared_command = (
                "python",
                "-m",
                "largerlm",
                "inspect-prepared",
                str(prepared_dir),
                "--require-prepared-memory-profile",
                "--require-glm-4bit",
                "--require-public-glm-5-2-shape",
                "--check-prompt-tokens",
                "1",
                "--check-max-new-tokens",
                "1",
                "--check-runtime-preflight",
                "--run-mpp-probe",
                "--run-mpsgraph-probe",
                "--prefill-backend-probe-timeout-seconds",
                GLM_5_2_BRINGUP_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
                "--write-launch-profile",
                str(launch_profile_path),
                "--json",
            )
            launch_audit_command = (
                "python",
                "-m",
                "largerlm",
                "inspect-prepared",
                str(prepared_dir),
                "--apply-launch-profile",
                str(launch_profile_path),
                "--lock-launch-profile",
                "--require-locked-launch-profile",
                "--require-prepared-memory-profile",
                "--require-glm-4bit",
                "--require-public-glm-5-2-shape",
                *GLM_5_2_BRINGUP_NON_ACCELERATED_PREFILL_AUDIT_FLAGS,
                *GLM_5_2_BRINGUP_PREFILL_ACCELERATION_FLAGS,
                *GLM_5_2_BRINGUP_PREFILL_GUARD_FLAGS,
                "--check-prompt-tokens",
                GLM_5_2_BRINGUP_LAUNCH_AUDIT_PROMPT_TOKENS,
                "--check-max-new-tokens",
                GLM_5_2_BRINGUP_MAX_NEW_TOKENS,
                *GLM_5_2_BRINGUP_CHECK_METAL_FINAL_LOGITS_FLAGS,
                "--check-runtime-preflight",
                "--run-mpp-probe",
                "--run-mpsgraph-probe",
                "--prefill-backend-probe-timeout-seconds",
                GLM_5_2_BRINGUP_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
                "--require-launch-audit",
                "--write-launch-audit",
                str(launch_audit_path),
                "--json",
            )
            minimal_smoke_command = (
                "python",
                "-m",
                "largerlm",
                "generate-prepared-token-ids",
                str(prepared_dir),
                "--apply-launch-profile",
                str(launch_profile_path),
                "--lock-launch-profile",
                "--require-locked-launch-profile",
                "--require-launch-audit",
                str(launch_audit_path),
                "--require-prepared-memory-profile",
                "--require-glm-4bit",
                "--require-public-glm-5-2-shape",
                *GLM_5_2_BRINGUP_NON_ACCELERATED_PREFILL_AUDIT_FLAGS,
                *GLM_5_2_BRINGUP_PREFILL_ACCELERATION_FLAGS,
                *GLM_5_2_BRINGUP_PREFILL_GUARD_FLAGS,
                *GLM_5_2_BRINGUP_PREFILL_GENERATION_ONLY_FLAGS,
                *GLM_5_2_BRINGUP_METAL_FINAL_LOGITS_FLAGS,
                "--run-mpp-probe",
                "--run-mpsgraph-probe",
                "--prefill-backend-probe-timeout-seconds",
                GLM_5_2_BRINGUP_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
                "--prompt-token-ids",
                "0",
                "--max-new-tokens",
                GLM_5_2_BRINGUP_MAX_NEW_TOKENS,
                "--write-result",
                str(minimal_smoke_result_path),
                "--quiet-runner",
                "--json",
            )

    metadata_ready_reason = None
    if not config_present:
        metadata_ready_reason = "config.json is missing"
    elif not can_run_metadata_preflight:
        metadata_ready_reason = "safetensors metadata is not ready"
    prepare_execute_blocked = _prepare_execute_blocked_reason(
        can_attempt_prepare=can_attempt_prepare,
        artifact_clean=artifact_clean,
        header_manifest_valid=header_manifest_valid,
        local_headers_ok=local_headers_ok,
    )
    prepared_manifest_path = prepared_dir / "manifest.json"
    prepared_manifest_present = prepared_manifest_path.exists()
    prepared_manifest_valid, prepared_manifest_error = _prepared_manifest_json_check(
        prepared_manifest_path
    )
    launch_profile_present = launch_profile_path.exists()
    launch_profile_valid, launch_profile_error = _launch_profile_json_check(
        launch_profile_path
    )
    launch_audit_present = launch_audit_path.exists()
    launch_audit_valid, launch_audit_error = _launch_audit_json_check(
        launch_audit_path
    )
    preflight_report_present = preflight_report_path.exists()
    preflight_report_valid, preflight_report_error = _preflight_report_json_check(
        preflight_report_path,
        model_dir=root,
    )
    prepare_dry_run_report_present = prepare_dry_run_report_path.exists()
    prepare_dry_run_report_valid, prepare_dry_run_report_error = (
        _prepare_dry_run_report_json_check(
            prepare_dry_run_report_path,
            model_dir=root,
            output_dir=prepared_dir,
        )
    )
    prefill_backend_report_present = prefill_backend_report_path.exists()
    prefill_backend_report_valid, prefill_backend_report_error = (
        _prefill_backend_report_json_check(prefill_backend_report_path)
    )
    minimal_smoke_result_present = minimal_smoke_result_path.exists()
    minimal_smoke_result_valid, minimal_smoke_result_error = (
        _minimal_smoke_result_json_check(
            minimal_smoke_result_path,
            prepared_manifest_path=prepared_manifest_path,
            launch_profile_path=launch_profile_path,
            launch_audit_path=launch_audit_path,
        )
    )
    post_copy_check_complete = bool(
        download_complete_proven
        and artifact_clean
        and (not header_manifest_valid or local_headers_ok is True)
    )
    bringup_plan = (
        _step(
            "fetch_headers",
            "Fetch safetensors headers only",
            header_fetch_command,
            blocked_reason=(
                "source repo is unavailable"
                if source_repo is None
                else "header manifest is already valid"
                if header_manifest_valid
                else None
            ),
            completed=header_manifest_valid,
            writes_artifacts=True,
        ),
        _step(
            "download_precheck",
            "Check remaining download disk budget",
            download_precheck_command,
            blocked_reason=(
                "download disk budget failed"
                if download_disk_ok is False
                else None
                if download_precheck_command is not None
                else "checkpoint metadata is unavailable"
            ),
            completed=download_disk_ok is True,
        ),
        _step(
            "prefill_backend_probe",
            "Probe M5 prefill acceleration backend",
            prefill_backend_command,
            blocked_reason=None,
            completed=prefill_backend_report_valid,
            prerequisite_step_ids=("download_precheck",),
            writes_artifacts=True,
        ),
        _step(
            "preflight",
            "Run GLM-5.2 metadata and budget preflight",
            preflight_command,
            blocked_reason=metadata_ready_reason,
            completed=prepared_manifest_valid or preflight_report_valid,
            prerequisite_step_ids=("prefill_backend_probe",),
            writes_artifacts=True,
        ),
        _step(
            "prepare_dry_run",
            "Dry-run packed layout and cache budget",
            prepare_dry_run_command,
            blocked_reason=metadata_ready_reason,
            completed=prepared_manifest_valid or prepare_dry_run_report_valid,
            prerequisite_step_ids=("preflight",),
            writes_artifacts=True,
        ),
        _step(
            "download_weights",
            "Download or resume checkpoint shards",
            hf_download_command,
            blocked_reason=(
                "source repo is unavailable"
                if hf_download_command is None
                else "download disk budget failed"
                if download_disk_ok is False
                else "download is already manifest-proven complete"
                if download_complete_proven
                else "download disk budget is not proven ok"
                if download_disk_ok is not True
                else None
            ),
            completed=download_complete_proven,
            prerequisite_step_ids=("prepare_dry_run",),
            writes_artifacts=True,
        ),
        _step(
            "post_copy_check",
            "Verify copied local shards without tensor payload reads",
            post_copy_check_command,
            blocked_reason=(
                "download is not manifest-proven complete"
                if not download_complete_proven
                else None
                if post_copy_check_command is not None
                else "checkpoint metadata is unavailable"
            ),
            completed=post_copy_check_complete,
            prerequisite_step_ids=("download_weights",),
        ),
        _step(
            "prepare_execute",
            "Write packed LargerLM artifacts",
            prepare_execute_command,
            blocked_reason=prepare_execute_blocked,
            completed=prepared_manifest_valid,
            prerequisite_step_ids=("post_copy_check",),
            reads_weight_payloads=True,
            writes_artifacts=True,
        ),
        _step(
            "inspect_prepared",
            "Inspect prepared package and write launch profile",
            inspect_prepared_command,
            blocked_reason=(
                None
                if inspect_prepared_command is not None
                else "prepare execute command is not available"
            ),
            completed=prepared_manifest_valid and launch_profile_valid,
            prerequisite_step_ids=("prepare_execute",),
        ),
        _step(
            "launch_audit",
            "Lock launch profile and write launch audit",
            launch_audit_command,
            blocked_reason=(
                None
                if launch_audit_command is not None
                else "inspect prepared command is not available"
            ),
            completed=(
                prepared_manifest_valid
                and launch_profile_valid
                and launch_audit_valid
            ),
            prerequisite_step_ids=("inspect_prepared",),
            writes_artifacts=True,
        ),
        _step(
            "minimal_smoke",
            "Run first audited 1-token prepared generation",
            minimal_smoke_command,
            blocked_reason=(
                None
                if minimal_smoke_command is not None
                else "launch audit command is not available"
            ),
            completed=(
                prepared_manifest_valid
                and launch_profile_valid
                and launch_audit_valid
                and minimal_smoke_result_valid
            ),
            prerequisite_step_ids=("launch_audit",),
            reads_weight_payloads=True,
            runs_model=True,
        ),
    )
    next_bringup_step = next(
        (step for step in bringup_plan if step.command_available),
        None,
    )

    return CheckpointArtifactStatus(
        model_dir=root,
        config_present=config_present,
        index_present=index_present,
        header_manifest_present=header_manifest_present,
        header_manifest_valid=header_manifest_valid,
        header_manifest_error=header_manifest_error,
        source_repo=source_repo,
        source_revision=revision_value if source_repo is not None else source_revision,
        source_endpoint=endpoint_value if source_repo is not None else source_endpoint,
        expected_tensor_bytes=expected_tensor_bytes,
        expected_safetensors_file_bytes=expected_file_bytes,
        remaining_safetensors_file_bytes=remaining_safetensors_file_bytes,
        present_safetensors_file_bytes=present_file_bytes,
        download_disk_budget=download_disk_budget,
        download_disk_ok=download_disk_ok,
        local_header_check_requested=bool(verify_local_headers),
        local_header_checked_shard_count=local_header_checked_count,
        local_header_ok_shard_count=local_header_ok_count,
        local_header_error_count=local_header_error_count,
        local_headers_ok=local_headers_ok,
        artifact_clean=artifact_clean,
        expected_shard_count=len(expected_shards),
        present_shard_count=present_shards,
        complete_shard_count=complete_count,
        missing_shard_count=missing_count,
        partial_shard_count=partial_count,
        extra_shard_count=len(extra_names),
        download_complete=download_complete,
        download_complete_proven=download_complete_proven,
        can_run_metadata_preflight=can_run_metadata_preflight,
        can_attempt_prepare=can_attempt_prepare,
        hf_download_command=hf_download_command,
        header_fetch_command=header_fetch_command,
        download_precheck_command=download_precheck_command,
        post_copy_check_command=post_copy_check_command,
        prefill_backend_command=prefill_backend_command,
        preflight_command=preflight_command,
        prepare_dry_run_command=prepare_dry_run_command,
        prepare_execute_command=prepare_execute_command,
        inspect_prepared_command=inspect_prepared_command,
        launch_audit_command=launch_audit_command,
        minimal_smoke_command=minimal_smoke_command,
        prepared_manifest_present=prepared_manifest_present,
        prepared_manifest_valid=prepared_manifest_valid,
        prepared_manifest_error=prepared_manifest_error,
        launch_profile_present=launch_profile_present,
        launch_profile_valid=launch_profile_valid,
        launch_profile_error=launch_profile_error,
        launch_audit_present=launch_audit_present,
        launch_audit_valid=launch_audit_valid,
        launch_audit_error=launch_audit_error,
        minimal_smoke_result_present=minimal_smoke_result_present,
        minimal_smoke_result_valid=minimal_smoke_result_valid,
        minimal_smoke_result_error=minimal_smoke_result_error,
        prefill_backend_report_present=prefill_backend_report_present,
        prefill_backend_report_valid=prefill_backend_report_valid,
        prefill_backend_report_error=prefill_backend_report_error,
        preflight_report_present=preflight_report_present,
        preflight_report_valid=preflight_report_valid,
        preflight_report_error=preflight_report_error,
        prepare_dry_run_report_present=prepare_dry_run_report_present,
        prepare_dry_run_report_valid=prepare_dry_run_report_valid,
        prepare_dry_run_report_error=prepare_dry_run_report_error,
        bringup_plan=bringup_plan,
        next_bringup_step=next_bringup_step,
        issues=tuple(issues),
        shards=tuple(shard_statuses),
        extra_shards=extra_names,
    )
