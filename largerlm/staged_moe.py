from __future__ import annotations

from array import array
import json
import math
import os
import shutil
import subprocess
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from .expert_io import (
    BatchExpertStageResult,
    ExpertIOPlanError,
    StaticExpertCapacityPlan,
    StaticExpertOverflow,
    StaticExpertTokenSlot,
    StaticExpertUsage,
    plan_batch_expert_io_tiles,
    stage_batch_experts,
    static_expert_capacity_binary_bytes,
    static_expert_capacity_json_bytes,
    validate_static_expert_capacity_binary,
    write_static_expert_capacity_binary,
    write_static_expert_capacity_plan,
)
from .safety import disk_budget

MoETokenBlock = Union[int, str]
MoEOutputAccumulator = str
MOE_BATCH_ACCUMULATOR_ENV = "LARGERLM_MOE_BATCH_ACCUMULATOR"


class StagedMoEError(RuntimeError):
    """Raised when a staged routed MoE batch cannot be executed safely."""


def _integer_value(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise StagedMoEError(f"{label} must be an integer")
    if type(value) is not int:
        raise StagedMoEError(f"{label} must be an integer")
    return int(value)


def _nonnegative_integer_value(value: object, *, label: str) -> int:
    parsed = _integer_value(value, label=label)
    if parsed < 0:
        raise StagedMoEError(f"{label} must be non-negative")
    return parsed


def _positive_integer_value(value: object, *, label: str) -> int:
    parsed = _integer_value(value, label=label)
    if parsed <= 0:
        raise StagedMoEError(f"{label} must be positive")
    return parsed


def _finite_float_value(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise StagedMoEError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise StagedMoEError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise StagedMoEError(f"{label} must be finite")
    return parsed


def _positive_float_value(value: object, *, label: str) -> float:
    parsed = _finite_float_value(value, label=label)
    if parsed <= 0:
        raise StagedMoEError(f"{label} must be positive")
    return parsed


@dataclass(frozen=True)
class StagedRoutedMoEBatchResult:
    runner_path: Path
    stage_manifest_path: Path
    stage_file_path: Path
    compact_layout_path: Path
    compact_layer_path: Path
    compact_routes_path: Path
    static_capacity_path: Path | None
    static_capacity_binary_path: Path | None
    input_path: Path
    output_path: Path
    output_dir: Path
    layer: int
    batch_tokens: int
    hidden_dim: int
    selected_experts: tuple[int, ...]
    input_bytes: int
    output_bytes: int
    token_bytes: int
    compact_stage_bytes: int
    compact_stage_materialized_bytes: int
    compact_stage_storage: str
    max_compact_stage_bytes: int
    stage_io_summary_available: bool
    stage_read_advice_available: bool
    stage_serial_read_bytes: int
    stage_unique_requested_bytes: int
    stage_planned_read_bytes: int
    stage_staged_bytes: int
    stage_waste_bytes: int
    stage_coalesced_savings_bytes: int
    stage_raw_range_count: int
    stage_coalesced_range_count: int
    stage_assignment_read_amplification: float
    stage_unique_read_amplification: float
    stage_staged_unique_read_amplification: float
    stage_budget_utilization: float
    stage_read_advice_supported: bool | None
    stage_read_advice_attempted_ranges: int
    stage_read_advice_calls: int
    stage_read_advice_bytes: int
    stage_read_advice_failures: int
    stage_read_advice_error: str | None
    copy_chunk_bytes: int
    stage_copy_seconds_ok: bool | None
    static_capacity_per_expert: int | None
    static_capacity_used_slots: int
    static_capacity_total_slots: int
    static_capacity_overflow_assignments: int
    static_capacity_binary_bytes: int
    moe_token_block: MoETokenBlock
    moe_token_block_mode: str | None
    effective_moe_token_block: int | None
    moe_max_expert_tokens: int | None
    moe_batch_buffer_bytes: int | None
    moe_estimated_peak_bytes: int | None
    moe_mxfp4_token_tile: int | None
    moe_mxfp4_vector_swiglu: bool | None
    moe_mxfp4_group32_specialized: bool | None
    moe_mxfp4_swiglu_activation: str | None
    moe_output_accumulator: str | None
    moe_output_accumulator_bytes: int | None
    command_count: int
    first_command: tuple[str, ...]
    stage_copy_elapsed_seconds: float | None = None
    stage_copy_throughput_gib_per_second: float | None = None
    moe_timing_sort_seconds: float | None = None
    moe_timing_setup_seconds: float | None = None
    moe_timing_expert_read_seconds: float | None = None
    moe_timing_input_read_seconds: float | None = None
    moe_timing_output_read_seconds: float | None = None
    moe_timing_kernel_seconds: float | None = None
    moe_timing_mxfp4_swiglu_kernel_seconds: float | None = None
    moe_timing_mxfp4_down_add_kernel_seconds: float | None = None
    moe_timing_output_write_seconds: float | None = None
    moe_timing_final_read_seconds: float | None = None
    moe_timing_total_seconds: float | None = None
    wall_total_elapsed_seconds: float | None = None
    wall_compact_stage_elapsed_seconds: float | None = None
    wall_routes_elapsed_seconds: float | None = None
    wall_static_capacity_elapsed_seconds: float | None = None
    wall_runner_elapsed_seconds: float | None = None
    wall_output_validation_elapsed_seconds: float | None = None


@dataclass(frozen=True)
class TiledStagedRoutedMoEBatchResult:
    runner_path: Path
    expert_layout_path: Path
    input_path: Path
    output_path: Path
    output_dir: Path
    layer: int
    batch_tokens: int
    hidden_dim: int
    tile_count: int
    tile_original_token_indices: tuple[tuple[int, ...], ...]
    tile_stage_results: tuple[BatchExpertStageResult, ...]
    tile_results: tuple[StagedRoutedMoEBatchResult, ...]
    total_stage_planned_read_bytes: int
    max_tile_stage_planned_read_bytes: int
    total_compact_stage_bytes: int
    max_tile_compact_stage_bytes: int
    output_bytes: int
    wall_total_elapsed_seconds: float | None = None
    wall_plan_elapsed_seconds: float | None = None
    wall_tile_router_elapsed_seconds: float | None = None
    wall_tile_input_elapsed_seconds: float | None = None
    wall_stage_elapsed_seconds: float | None = None
    wall_staged_moe_elapsed_seconds: float | None = None
    wall_scatter_elapsed_seconds: float | None = None
    wall_cleanup_elapsed_seconds: float | None = None
    wall_output_validation_elapsed_seconds: float | None = None


@dataclass(frozen=True)
class StagedRoutedMoEBatchPlanJob:
    layout_path: Path
    layer: int
    routes_json_path: Path | None
    routes_bin_path: Path | None
    input_path: Path
    output_path: Path
    batch_tokens: int
    max_k: int
    max_slot_mib: int
    max_runner_scratch_mib: int
    moe_token_block: MoETokenBlock = "auto"
    expert_read_advise_merge_gap_bytes: int = 0
    expert_read_advise_align_bytes: int = 0


@dataclass(frozen=True)
class StagedRoutedMoEBatchPlanResult:
    runner_path: Path
    plan_path: Path
    job_count: int
    command_count: int
    first_command: tuple[str, ...]
    wall_runner_elapsed_seconds: float
    runner_reported_elapsed_seconds: float | None
    output_paths: tuple[Path, ...]
    runner_stdout: str | None = None


@dataclass(frozen=True)
class StagedRoutedMoEBatchPlanServerResult:
    runner_path: Path
    plan_paths: tuple[Path, ...]
    plan_count: int
    job_count: int
    command_count: int
    first_command: tuple[str, ...]
    wall_runner_elapsed_seconds: float
    output_paths: tuple[Path, ...]


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise StagedMoEError(f"failed to read staged MoE JSON {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StagedMoEError(f"failed to parse staged MoE JSON {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StagedMoEError(f"staged MoE JSON {p} must be an object")
    return payload


def _require_static_capacity_disk_budget(
    *,
    json_path: Path | None,
    json_bytes: int,
    binary_path: Path,
    binary_bytes: int,
    safety_margin_bytes: int,
) -> None:
    requirements: dict[Path, int] = {}
    if json_path is not None and json_bytes > 0:
        requirements[json_path.parent] = (
            requirements.get(json_path.parent, 0) + json_bytes
        )
    requirements[binary_path.parent] = (
        requirements.get(binary_path.parent, 0) + binary_bytes
    )
    for output_dir, required_bytes in requirements.items():
        budget = disk_budget(
            output_dir,
            required_bytes,
            safety_margin_bytes=safety_margin_bytes,
        )
        if not budget.ok:
            raise StagedMoEError(
                "not enough free disk for static capacity route artifacts: "
                f"need {required_bytes + safety_margin_bytes} bytes including margin, "
                f"have {budget.available_bytes} bytes"
            )


def _normalize_moe_token_block(value: MoETokenBlock) -> tuple[str, MoETokenBlock]:
    if isinstance(value, bool):
        raise StagedMoEError("moe_token_block must be positive or 'auto'")
    if isinstance(value, int):
        if value <= 0:
            raise StagedMoEError("moe_token_block must be positive or 'auto'")
        return str(value), value
    text = str(value).strip().lower()
    if text == "auto":
        return "auto", "auto"
    try:
        parsed = int(text)
    except ValueError as exc:
        raise StagedMoEError("moe_token_block must be positive or 'auto'") from exc
    if parsed <= 0:
        raise StagedMoEError("moe_token_block must be positive or 'auto'")
    return str(parsed), parsed


def _normalize_moe_output_accumulator(value: object) -> MoEOutputAccumulator:
    if value is None:
        return "env"
    text = str(value).strip().lower()
    if text in {"", "env", "inherit", "default", "auto"}:
        return "env"
    if text in {"file", "disk"}:
        return "file"
    if text in {"memory", "mem", "in-memory", "ram"}:
        return "memory"
    raise StagedMoEError("moe_output_accumulator must be env, file, or memory")


def _moe_output_accumulator_env(
    moe_output_accumulator: MoEOutputAccumulator,
) -> dict[str, str] | None:
    mode = _normalize_moe_output_accumulator(moe_output_accumulator)
    if mode == "env":
        return None
    env = dict(os.environ)
    env[MOE_BATCH_ACCUMULATOR_ENV] = mode
    return env


def _find_layer(layout: dict[str, Any], layer_id: int) -> dict[str, Any]:
    layer_id = _integer_value(layer_id, label="layer")
    layers = layout.get("layers")
    if not isinstance(layers, list):
        raise StagedMoEError("expert layout missing layers array")
    for layer in layers:
        if (
            isinstance(layer, dict)
            and type(layer.get("layer")) is int
            and layer.get("layer") == layer_id
        ):
            return layer
    raise StagedMoEError(f"layer {layer_id} not found in expert layout")


def _component_shape2(layer: dict[str, Any], component_name: str) -> tuple[int, int]:
    components = layer.get("components")
    if not isinstance(components, list):
        raise StagedMoEError("expert layer missing components array")
    for component in components:
        if isinstance(component, dict) and component.get("name") == component_name:
            shape = component.get("shape")
            if (
                not isinstance(shape, list)
                or len(shape) < 2
                or type(shape[0]) is not int
                or type(shape[1]) is not int
            ):
                raise StagedMoEError(f"component {component_name} must have shape [rows, cols]")
            return int(shape[0]), int(shape[1])
    raise StagedMoEError(f"expert layout missing component {component_name}")


def _copy_exact_range(
    *,
    source,
    destination,
    source_offset: int,
    length: int,
    copy_chunk_bytes: int,
    copy_buffer: bytearray | None = None,
) -> None:
    if copy_chunk_bytes <= 0:
        raise StagedMoEError("copy_chunk_bytes must be positive")
    remaining = length
    current_offset = source_offset
    buffer = copy_buffer
    if buffer is None or len(buffer) < min(copy_chunk_bytes, max(remaining, 1)):
        buffer = bytearray(min(copy_chunk_bytes, max(remaining, 1)))
    view = memoryview(buffer)
    while remaining:
        chunk_size = min(copy_chunk_bytes, remaining)
        chunk_view = view[:chunk_size]
        if hasattr(os, "preadv"):
            read = os.preadv(source.fileno(), [chunk_view], current_offset)
        else:
            chunk = os.pread(source.fileno(), chunk_size, current_offset)
            read = len(chunk)
            if read == chunk_size:
                chunk_view[:] = chunk
        if read != chunk_size:
            raise StagedMoEError(
                f"failed to read {chunk_size} compact stage bytes at source offset "
                f"{current_offset}"
            )
        written = destination.write(chunk_view)
        if written != chunk_size:
            raise StagedMoEError(
                f"failed to write {chunk_size} compact stage bytes at source offset "
                f"{current_offset}"
            )
        remaining -= chunk_size
        current_offset += chunk_size


def _remove_partial_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        _remove_partial_file(tmp_path)
        raise


def _run_command(
    cmd: list[str],
    *,
    echo_output: bool,
    env: dict[str, str] | None = None,
) -> str:
    try:
        completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
    except OSError as exc:
        raise StagedMoEError(f"failed to launch staged routed MoE command: {exc}") from exc
    if echo_output:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
    if completed.returncode != 0:
        detail = ""
        if completed.stdout:
            detail += f"\nstdout:\n{completed.stdout[-4000:]}"
        if completed.stderr:
            detail += f"\nstderr:\n{completed.stderr[-4000:]}"
        raise StagedMoEError(
            f"staged routed MoE command failed with exit {completed.returncode}: "
            f"{' '.join(cmd)}{detail}"
        )
    return completed.stdout


def _plan_path(path: str | Path, *, label: str) -> Path:
    value = Path(path)
    if str(value) == "":
        raise StagedMoEError(f"{label} must not be empty")
    return value


def _plan_job_payload(job: StagedRoutedMoEBatchPlanJob) -> dict[str, Any]:
    layer = _nonnegative_integer_value(job.layer, label="plan job layer")
    batch_tokens = _positive_integer_value(
        job.batch_tokens,
        label="plan job batch_tokens",
    )
    max_k = _positive_integer_value(job.max_k, label="plan job max_k")
    if max_k > 64:
        raise StagedMoEError("plan job max_k must be in 1..64")
    max_slot_mib = _positive_integer_value(
        job.max_slot_mib,
        label="plan job max_slot_mib",
    )
    max_runner_scratch_mib = _positive_integer_value(
        job.max_runner_scratch_mib,
        label="plan job max_runner_scratch_mib",
    )
    merge_gap_bytes = _nonnegative_integer_value(
        job.expert_read_advise_merge_gap_bytes,
        label="plan job expert_read_advise_merge_gap_bytes",
    )
    align_bytes = _nonnegative_integer_value(
        job.expert_read_advise_align_bytes,
        label="plan job expert_read_advise_align_bytes",
    )
    _moe_token_block_arg, moe_token_block_value = _normalize_moe_token_block(
        job.moe_token_block
    )
    if (job.routes_json_path is None) == (job.routes_bin_path is None):
        raise StagedMoEError(
            "plan job must include exactly one of routes_json_path or routes_bin_path"
        )
    payload: dict[str, Any] = {
        "layout": str(_plan_path(job.layout_path, label="plan job layout_path")),
        "layer": layer,
        "input_f32": str(_plan_path(job.input_path, label="plan job input_path")),
        "output_f32": str(_plan_path(job.output_path, label="plan job output_path")),
        "batch_tokens": batch_tokens,
        "max_k": max_k,
        "max_slot_mib": max_slot_mib,
        "max_runner_scratch_mib": max_runner_scratch_mib,
        "moe_token_block": moe_token_block_value,
        "expert_read_advise_merge_gap_bytes": merge_gap_bytes,
        "expert_read_advise_align_bytes": align_bytes,
    }
    if job.routes_json_path is not None:
        payload["routes_json"] = str(
            _plan_path(job.routes_json_path, label="plan job routes_json_path")
        )
    else:
        payload["routes_bin"] = str(
            _plan_path(job.routes_bin_path, label="plan job routes_bin_path")
        )
    return payload


def write_staged_routed_moe_batch_plan(
    plan_path: str | Path,
    jobs: tuple[StagedRoutedMoEBatchPlanJob, ...] | list[StagedRoutedMoEBatchPlanJob],
) -> Path:
    if not isinstance(jobs, (tuple, list)) or not jobs:
        raise StagedMoEError("staged routed MoE batch plan requires at least one job")
    path = _plan_path(plan_path, label="plan_path")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "largerlm.staged_routed_moe_batch_plan.v1",
        "version": 1,
        "job_count": len(jobs),
        "jobs": [_plan_job_payload(job) for job in jobs],
    }
    try:
        _write_json_atomic(path, payload)
    except OSError as exc:
        raise StagedMoEError(f"failed to write staged MoE batch plan {path}: {exc}") from exc
    return path


def run_staged_routed_moe_batch_plan(
    *,
    runner_path: str | Path,
    plan_path: str | Path,
    echo_runner_output: bool = True,
    moe_output_accumulator: MoEOutputAccumulator = "env",
) -> StagedRoutedMoEBatchPlanResult:
    runner = _plan_path(runner_path, label="runner_path")
    if not runner.exists():
        raise StagedMoEError(f"runner not found: {runner}")
    plan = _plan_path(plan_path, label="plan_path")
    job_count, output_paths = _batch_plan_outputs(plan)
    cmd = [
        str(runner),
        "--run-moe-batch-plan",
        "--batch-plan-json",
        str(plan),
    ]
    started = time.perf_counter()
    runner_stdout = _run_command(
        cmd,
        echo_output=echo_runner_output,
        env=_moe_output_accumulator_env(moe_output_accumulator),
    )
    wall_runner_elapsed = time.perf_counter() - started
    _require_batch_plan_outputs(output_paths)
    return StagedRoutedMoEBatchPlanResult(
        runner_path=runner,
        plan_path=plan,
        job_count=job_count,
        command_count=1,
        first_command=tuple(cmd),
        wall_runner_elapsed_seconds=wall_runner_elapsed,
        runner_reported_elapsed_seconds=_runner_stat_float(
            runner_stdout,
            "plan timing total",
        ),
        output_paths=tuple(output_paths),
        runner_stdout=runner_stdout,
    )


def _batch_plan_outputs(plan: Path) -> tuple[int, list[Path]]:
    payload = _load_json(plan)
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise StagedMoEError("staged routed MoE batch plan must include jobs")
    output_paths: list[Path] = []
    for index, raw_job in enumerate(jobs):
        if not isinstance(raw_job, dict):
            raise StagedMoEError(f"staged routed MoE batch plan job {index} must be an object")
        raw_output = raw_job.get("output_f32")
        if not isinstance(raw_output, str) or not raw_output:
            raise StagedMoEError(
                f"staged routed MoE batch plan job {index} missing output_f32"
            )
        output_paths.append(Path(raw_output))
    return len(jobs), output_paths


def _require_batch_plan_outputs(output_paths: list[Path] | tuple[Path, ...]) -> None:
    for output_path in output_paths:
        if not output_path.exists():
            raise StagedMoEError(
                f"staged routed MoE batch plan did not write expected output {output_path}"
            )


def run_staged_routed_moe_batch_plan_server(
    *,
    runner_path: str | Path,
    plan_paths: tuple[str | Path, ...] | list[str | Path],
    echo_runner_output: bool = True,
    moe_output_accumulator: MoEOutputAccumulator = "env",
) -> StagedRoutedMoEBatchPlanServerResult:
    runner = _plan_path(runner_path, label="runner_path")
    if not runner.exists():
        raise StagedMoEError(f"runner not found: {runner}")
    if not isinstance(plan_paths, (tuple, list)) or not plan_paths:
        raise StagedMoEError("staged routed MoE batch plan server requires plans")
    plans = tuple(_plan_path(path, label="plan_path") for path in plan_paths)
    output_paths: list[Path] = []
    job_count = 0
    for plan in plans:
        plan_job_count, plan_outputs = _batch_plan_outputs(plan)
        job_count += plan_job_count
        output_paths.extend(plan_outputs)
    cmd = [str(runner), "--run-moe-batch-plan-server-jsonl"]
    request_lines = [
        json.dumps({"batch_plan_json": str(plan)}, separators=(",", ":"))
        for plan in plans
    ]
    request_lines.append(json.dumps({"command": "quit"}, separators=(",", ":")))
    request_payload = "\n".join(request_lines) + "\n"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            input=request_payload,
            text=True,
            capture_output=True,
            env=_moe_output_accumulator_env(moe_output_accumulator),
        )
    except OSError as exc:
        raise StagedMoEError(
            f"failed to launch staged routed MoE plan server: {exc}"
        ) from exc
    wall_runner_elapsed = time.perf_counter() - started
    if echo_runner_output:
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="")
    if completed.returncode != 0:
        detail = ""
        if completed.stdout:
            detail += f"\nstdout:\n{completed.stdout[-4000:]}"
        if completed.stderr:
            detail += f"\nstderr:\n{completed.stderr[-4000:]}"
        raise StagedMoEError(
            "staged routed MoE batch plan server failed with exit "
            f"{completed.returncode}: {' '.join(cmd)}{detail}"
        )
    _require_batch_plan_outputs(output_paths)
    return StagedRoutedMoEBatchPlanServerResult(
        runner_path=runner,
        plan_paths=plans,
        plan_count=len(plans),
        job_count=job_count,
        command_count=1,
        first_command=tuple(cmd),
        wall_runner_elapsed_seconds=wall_runner_elapsed,
        output_paths=tuple(output_paths),
    )


class StagedRoutedMoEBatchPlanServerSession:
    def __init__(
        self,
        *,
        runner_path: str | Path,
        echo_runner_output: bool = True,
        moe_output_accumulator: MoEOutputAccumulator = "env",
    ) -> None:
        self.runner_path = _plan_path(runner_path, label="runner_path")
        if not self.runner_path.exists():
            raise StagedMoEError(f"runner not found: {self.runner_path}")
        self.echo_runner_output = echo_runner_output
        self.moe_output_accumulator = _normalize_moe_output_accumulator(
            moe_output_accumulator
        )
        self._process: subprocess.Popen[str] | None = None
        self._recent_output: list[str] = []

    def __enter__(self) -> "StagedRoutedMoEBatchPlanServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise StagedMoEError("staged routed MoE plan server is already started")
        cmd = [str(self.runner_path), "--run-moe-batch-plan-server-jsonl"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_moe_output_accumulator_env(self.moe_output_accumulator),
            )
        except OSError as exc:
            raise StagedMoEError(
                f"failed to launch staged routed MoE plan server: {exc}"
            ) from exc

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(
                    json.dumps({"command": "quit"}, separators=(",", ":")) + "\n"
                )
                process.stdin.flush()
            except OSError:
                pass
        self._drain_until(
            success_marker="LargerLM MoE batch plan server done",
            allow_eof=True,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._process = None

    def submit_plan(
        self,
        plan_path: str | Path,
    ) -> StagedRoutedMoEBatchPlanResult:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise StagedMoEError("staged routed MoE plan server stdin is unavailable")
        if process.poll() is not None:
            raise StagedMoEError(
                "staged routed MoE plan server is not running"
                + self._recent_output_detail()
            )
        plan = _plan_path(plan_path, label="plan_path")
        job_count, output_paths = _batch_plan_outputs(plan)
        started = time.perf_counter()
        try:
            process.stdin.write(
                json.dumps({"batch_plan_json": str(plan)}, separators=(",", ":"))
                + "\n"
            )
            process.stdin.flush()
        except OSError as exc:
            raise StagedMoEError(
                f"failed to send staged routed MoE plan to server: {exc}"
            ) from exc
        runner_stdout = self._drain_until(success_marker="  server request:      ok")
        wall_elapsed = time.perf_counter() - started
        _require_batch_plan_outputs(output_paths)
        return StagedRoutedMoEBatchPlanResult(
            runner_path=self.runner_path,
            plan_path=plan,
            job_count=job_count,
            command_count=0,
            first_command=(str(self.runner_path), "--run-moe-batch-plan-server-jsonl"),
            wall_runner_elapsed_seconds=wall_elapsed,
            runner_reported_elapsed_seconds=_runner_stat_float(
                runner_stdout,
                "plan timing total",
            ),
            output_paths=tuple(output_paths),
            runner_stdout=runner_stdout,
        )

    def _drain_until(self, *, success_marker: str, allow_eof: bool = False) -> str:
        process = self._process
        if process is None or process.stdout is None:
            return ""
        captured: list[str] = []
        while True:
            line = process.stdout.readline()
            if line == "":
                if allow_eof:
                    return "".join(captured)
                status = process.poll()
                if status is None:
                    raise StagedMoEError(
                        "staged routed MoE plan server stdout closed unexpectedly"
                        + self._recent_output_detail()
                    )
                raise StagedMoEError(
                    f"staged routed MoE plan server exited with {status}"
                    + self._recent_output_detail()
                )
            captured.append(line)
            self._record_output(line)
            if self.echo_runner_output:
                print(line, end="")
            if "  server request:      failed" in line:
                raise StagedMoEError(
                    "staged routed MoE plan server request failed"
                    + self._recent_output_detail()
                )
            if success_marker in line:
                return "".join(captured)

    def _record_output(self, line: str) -> None:
        self._recent_output.append(line)
        if len(self._recent_output) > 120:
            del self._recent_output[: len(self._recent_output) - 120]

    def _recent_output_detail(self) -> str:
        if not self._recent_output:
            return ""
        return "\nrecent output:\n" + "".join(self._recent_output[-40:])


def _normalize_row_indices(
    token_indices: tuple[int, ...] | list[int],
    *,
    batch_tokens: int,
) -> tuple[int, ...]:
    if batch_tokens <= 0:
        raise StagedMoEError("batch_tokens must be positive")
    seen: set[int] = set()
    parsed: list[int] = []
    for raw in token_indices:
        index = _nonnegative_integer_value(raw, label="token_index")
        if index >= batch_tokens:
            raise StagedMoEError("token_index is outside batch range")
        if index in seen:
            raise StagedMoEError("token_indices must be unique")
        seen.add(index)
        parsed.append(index)
    if not parsed:
        raise StagedMoEError("token_indices must be non-empty")
    return tuple(parsed)


def _f32_row_bytes(hidden_dim: int) -> int:
    hidden_dim = _positive_integer_value(hidden_dim, label="hidden_dim")
    return hidden_dim * 4


def _contiguous_row_runs(indices: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    if not indices:
        return ()
    runs: list[tuple[int, int, int]] = []
    partial_start = 0
    source_start = indices[0]
    previous = source_start
    count = 1
    for partial_index, index in enumerate(indices[1:], start=1):
        if index == previous + 1:
            previous = index
            count += 1
            continue
        runs.append((partial_start, source_start, count))
        partial_start = partial_index
        source_start = index
        previous = index
        count = 1
    runs.append((partial_start, source_start, count))
    return tuple(runs)


def _add_f32_byte_chunks(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right) or len(left) % 4 != 0:
        raise StagedMoEError("f32 byte chunks must have equal 4-byte-aligned length")
    left_values = array("f")
    right_values = array("f")
    left_values.frombytes(left)
    right_values.frombytes(right)
    if sys.byteorder != "little":
        left_values.byteswap()
        right_values.byteswap()
    for index, value in enumerate(right_values):
        left_values[index] += value
    if sys.byteorder != "little":
        left_values.byteswap()
    return left_values.tobytes()


def _gather_f32_rows(
    *,
    input_path: str | Path,
    output_path: str | Path,
    token_indices: tuple[int, ...] | list[int],
    batch_tokens: int,
    hidden_dim: int,
) -> Path:
    indices = _normalize_row_indices(token_indices, batch_tokens=batch_tokens)
    row_bytes = _f32_row_bytes(hidden_dim)
    input_p = Path(input_path)
    output_p = Path(output_path)
    expected_input_bytes = batch_tokens * row_bytes
    try:
        input_bytes = input_p.stat().st_size
    except OSError as exc:
        raise StagedMoEError(f"failed to stat f32 row source {input_p}: {exc}") from exc
    if input_bytes != expected_input_bytes:
        raise StagedMoEError(
            f"f32 row source has {input_bytes} bytes, expected "
            f"{expected_input_bytes}"
        )
    output_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_p.with_name(output_p.name + ".tmp")
    try:
        with input_p.open("rb") as source, tmp_path.open("wb") as destination:
            for _partial_start, source_start, count in _contiguous_row_runs(indices):
                run_bytes = count * row_bytes
                source.seek(source_start * row_bytes)
                chunk = source.read(run_bytes)
                if len(chunk) != run_bytes:
                    raise StagedMoEError("f32 row source ended early")
                destination.write(chunk)
        tmp_path.replace(output_p)
    except (OSError, StagedMoEError) as exc:
        _remove_partial_file(tmp_path)
        raise StagedMoEError(f"failed to gather f32 rows: {exc}") from exc
    return output_p


def _zero_f32_file(path: Path, *, batch_tokens: int, hidden_dim: int) -> None:
    row_bytes = _f32_row_bytes(hidden_dim)
    zero_row = b"\0" * row_bytes
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("wb") as handle:
            for _ in range(batch_tokens):
                handle.write(zero_row)
        tmp_path.replace(path)
    except OSError as exc:
        _remove_partial_file(tmp_path)
        raise StagedMoEError(f"failed to initialize f32 accumulator {path}: {exc}") from exc


def _scatter_add_f32_rows(
    *,
    accumulator_path: str | Path,
    partial_path: str | Path,
    token_indices: tuple[int, ...] | list[int],
    batch_tokens: int,
    hidden_dim: int,
    initialize: bool = False,
) -> Path:
    if type(initialize) is not bool:
        raise StagedMoEError("initialize must be a boolean")
    indices = _normalize_row_indices(token_indices, batch_tokens=batch_tokens)
    row_bytes = _f32_row_bytes(hidden_dim)
    accumulator = Path(accumulator_path)
    partial = Path(partial_path)
    if initialize:
        _zero_f32_file(accumulator, batch_tokens=batch_tokens, hidden_dim=hidden_dim)
    expected_accumulator_bytes = batch_tokens * row_bytes
    expected_partial_bytes = len(indices) * row_bytes
    try:
        accumulator_bytes = accumulator.stat().st_size
        partial_bytes = partial.stat().st_size
    except OSError as exc:
        raise StagedMoEError(f"failed to stat f32 scatter-add inputs: {exc}") from exc
    if accumulator_bytes != expected_accumulator_bytes:
        raise StagedMoEError(
            f"f32 accumulator has {accumulator_bytes} bytes, expected "
            f"{expected_accumulator_bytes}"
        )
    if partial_bytes != expected_partial_bytes:
        raise StagedMoEError(
            f"f32 partial has {partial_bytes} bytes, expected "
            f"{expected_partial_bytes}"
        )
    try:
        with accumulator.open("r+b") as acc, partial.open("rb") as part:
            for partial_start, token_index, count in _contiguous_row_runs(indices):
                run_bytes = count * row_bytes
                part.seek(partial_start * row_bytes)
                partial_chunk = part.read(run_bytes)
                if len(partial_chunk) != run_bytes:
                    raise StagedMoEError("f32 partial ended early")
                acc_offset = token_index * row_bytes
                acc.seek(acc_offset)
                accumulator_chunk = acc.read(run_bytes)
                if len(accumulator_chunk) != run_bytes:
                    raise StagedMoEError("f32 accumulator ended early")
                acc.seek(acc_offset)
                acc.write(_add_f32_byte_chunks(accumulator_chunk, partial_chunk))
    except (OSError, struct.error, StagedMoEError) as exc:
        raise StagedMoEError(f"failed to scatter-add f32 rows: {exc}") from exc
    return accumulator


def _tile_token_routes_for_experts(
    token_routes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    selected_experts: tuple[int, ...],
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    if not selected_experts:
        raise StagedMoEError("selected_experts must be non-empty")
    selected = set(selected_experts)
    original_token_indices: list[int] = []
    tile_routes: list[dict[str, Any]] = []
    for route in token_routes:
        if not isinstance(route, dict):
            raise StagedMoEError("token route entries must be objects")
        original_token = _nonnegative_integer_value(
            route.get("token_index"),
            label="token route token_index",
        )
        experts = route.get("experts")
        weights = route.get("weights")
        if not isinstance(experts, list) or not isinstance(weights, list):
            raise StagedMoEError("token route must include experts and weights")
        if len(experts) != len(weights) or not experts:
            raise StagedMoEError("token route experts/weights are invalid")
        tile_experts: list[int] = []
        tile_weights: list[float] = []
        for raw_expert, raw_weight in zip(experts, weights):
            expert = _nonnegative_integer_value(
                raw_expert,
                label="token route expert",
            )
            weight = _finite_float_value(raw_weight, label="token route weight")
            if expert in selected:
                tile_experts.append(expert)
                tile_weights.append(weight)
        if tile_experts:
            original_token_indices.append(original_token)
            tile_routes.append(
                {
                    "token_index": len(tile_routes),
                    "experts": tile_experts,
                    "weights": tile_weights,
                }
            )
    return tuple(original_token_indices), tile_routes


def _write_tile_router_jsons(
    *,
    output_dir: str | Path,
    token_routes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    selected_experts: tuple[int, ...],
) -> tuple[tuple[int, ...], Path]:
    original_token_indices, tile_routes = _tile_token_routes_for_experts(
        token_routes,
        selected_experts=selected_experts,
    )
    if not tile_routes:
        raise StagedMoEError("expert tile has no active token routes")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, route in enumerate(tile_routes):
            _write_json_atomic(out_dir / f"token_{index:06d}.router.json", route)
    except OSError as exc:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise StagedMoEError(f"failed to write tile router JSONs: {exc}") from exc
    return original_token_indices, out_dir


def _runner_stat(stdout: str, label: str) -> str | None:
    prefix = f"{label}:"
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.split(":", 1)[1].strip()
    return None


def _runner_stat_int(stdout: str, label: str) -> int | None:
    value = _runner_stat(stdout, label)
    if value is None:
        return None
    first = value.split()[0]
    try:
        return int(first)
    except ValueError:
        return None


def _runner_stat_float(stdout: str, label: str) -> float | None:
    value = _runner_stat(stdout, label)
    if value is None:
        return None
    first = value.split()[0]
    try:
        result = float(first)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _runner_stat_bool(stdout: str, label: str) -> bool | None:
    value = _runner_stat(stdout, label)
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"yes", "true", "1", "on"}:
        return True
    if normalized in {"no", "false", "0", "off"}:
        return False
    return None


def _optional_nonnegative_manifest_int(
    payload: dict[str, Any],
    field: str,
    *,
    label: str,
) -> int:
    if field not in payload:
        return 0
    return _nonnegative_integer_value(payload.get(field), label=label)


def _optional_nonnegative_manifest_float(
    payload: dict[str, Any],
    field: str,
    *,
    label: str,
) -> float:
    if field not in payload:
        return 0.0
    value = _finite_float_value(payload.get(field), label=label)
    if value < 0.0:
        raise StagedMoEError(f"{label} must be non-negative")
    return value


def _optional_nonnegative_manifest_float_or_none(
    payload: dict[str, Any],
    field: str,
    *,
    label: str,
) -> float | None:
    if field not in payload:
        return None
    raw = payload.get(field)
    if raw is None:
        return None
    value = _finite_float_value(raw, label=label)
    if value < 0.0:
        raise StagedMoEError(f"{label} must be non-negative")
    return value


def _optional_manifest_bool_or_none(
    payload: dict[str, Any],
    field: str,
    *,
    label: str,
) -> bool | None:
    if field not in payload:
        return None
    value = payload.get(field)
    if value is None:
        return None
    if type(value) is not bool:
        raise StagedMoEError(f"{label} must be boolean")
    return value


def _stage_manifest_telemetry(manifest: dict[str, Any]) -> dict[str, object]:
    summary = manifest.get("io_summary")
    telemetry: dict[str, object] = {
        "stage_io_summary_available": False,
        "stage_read_advice_available": False,
        "stage_serial_read_bytes": 0,
        "stage_unique_requested_bytes": 0,
        "stage_planned_read_bytes": 0,
        "stage_staged_bytes": 0,
        "stage_waste_bytes": 0,
        "stage_coalesced_savings_bytes": 0,
        "stage_raw_range_count": 0,
        "stage_coalesced_range_count": 0,
        "stage_assignment_read_amplification": 0.0,
        "stage_unique_read_amplification": 0.0,
        "stage_staged_unique_read_amplification": 0.0,
        "stage_budget_utilization": 0.0,
        "stage_copy_elapsed_seconds": None,
        "stage_copy_throughput_gib_per_second": None,
        "stage_copy_seconds_ok": None,
        "stage_read_advice_supported": None,
        "stage_read_advice_attempted_ranges": 0,
        "stage_read_advice_calls": 0,
        "stage_read_advice_bytes": 0,
        "stage_read_advice_failures": 0,
        "stage_read_advice_error": None,
    }
    if summary is not None:
        if not isinstance(summary, dict):
            raise StagedMoEError("stage manifest io_summary must be an object")
        telemetry.update(
            {
                "stage_io_summary_available": True,
                "stage_serial_read_bytes": _optional_nonnegative_manifest_int(
                    summary,
                    "serial_read_bytes",
                    label="stage manifest io_summary serial_read_bytes",
                ),
                "stage_unique_requested_bytes": _optional_nonnegative_manifest_int(
                    summary,
                    "unique_requested_bytes",
                    label="stage manifest io_summary unique_requested_bytes",
                ),
                "stage_planned_read_bytes": _optional_nonnegative_manifest_int(
                    summary,
                    "planned_read_bytes",
                    label="stage manifest io_summary planned_read_bytes",
                ),
                "stage_staged_bytes": _optional_nonnegative_manifest_int(
                    summary,
                    "staged_bytes",
                    label="stage manifest io_summary staged_bytes",
                ),
                "stage_waste_bytes": _optional_nonnegative_manifest_int(
                    summary,
                    "waste_bytes",
                    label="stage manifest io_summary waste_bytes",
                ),
                "stage_coalesced_savings_bytes": _optional_nonnegative_manifest_int(
                    summary,
                    "coalesced_savings_bytes",
                    label="stage manifest io_summary coalesced_savings_bytes",
                ),
                "stage_raw_range_count": _optional_nonnegative_manifest_int(
                    summary,
                    "raw_range_count",
                    label="stage manifest io_summary raw_range_count",
                ),
                "stage_coalesced_range_count": _optional_nonnegative_manifest_int(
                    summary,
                    "coalesced_range_count",
                    label="stage manifest io_summary coalesced_range_count",
                ),
                "stage_assignment_read_amplification": (
                    _optional_nonnegative_manifest_float(
                        summary,
                        "assignment_read_amplification",
                        label=(
                            "stage manifest io_summary "
                            "assignment_read_amplification"
                        ),
                    )
                ),
                "stage_unique_read_amplification": (
                    _optional_nonnegative_manifest_float(
                        summary,
                        "unique_read_amplification",
                        label="stage manifest io_summary unique_read_amplification",
                    )
                ),
                "stage_staged_unique_read_amplification": (
                    _optional_nonnegative_manifest_float(
                        summary,
                        "staged_unique_read_amplification",
                        label=(
                            "stage manifest io_summary "
                            "staged_unique_read_amplification"
                        ),
                    )
                ),
                "stage_budget_utilization": _optional_nonnegative_manifest_float(
                    summary,
                    "stage_budget_utilization",
                    label="stage manifest io_summary stage_budget_utilization",
                ),
                "stage_copy_elapsed_seconds": (
                    _optional_nonnegative_manifest_float_or_none(
                        summary,
                        "copy_elapsed_seconds",
                        label="stage manifest io_summary copy_elapsed_seconds",
                    )
                ),
                "stage_copy_throughput_gib_per_second": (
                    _optional_nonnegative_manifest_float_or_none(
                        summary,
                        "copy_throughput_gib_per_second",
                        label=(
                            "stage manifest io_summary "
                            "copy_throughput_gib_per_second"
                        ),
                    )
                ),
                "stage_copy_seconds_ok": _optional_manifest_bool_or_none(
                    summary,
                    "copy_seconds_ok",
                    label="stage manifest io_summary copy_seconds_ok",
                ),
            }
        )

    read_advice = manifest.get("read_advice")
    if read_advice is not None:
        if not isinstance(read_advice, dict):
            raise StagedMoEError("stage manifest read_advice must be an object")
        supported = read_advice.get("supported")
        if type(supported) is not bool:
            raise StagedMoEError("stage manifest read_advice supported must be boolean")
        raw_error = read_advice.get("error")
        if raw_error is not None and not isinstance(raw_error, str):
            raise StagedMoEError("stage manifest read_advice error must be a string")
        telemetry.update(
            {
                "stage_read_advice_available": True,
                "stage_read_advice_supported": supported,
                "stage_read_advice_attempted_ranges": (
                    _optional_nonnegative_manifest_int(
                        read_advice,
                        "attempted_ranges",
                        label="stage manifest read_advice attempted_ranges",
                    )
                ),
                "stage_read_advice_calls": _optional_nonnegative_manifest_int(
                    read_advice,
                    "calls",
                    label="stage manifest read_advice calls",
                ),
                "stage_read_advice_bytes": _optional_nonnegative_manifest_int(
                    read_advice,
                    "advised_bytes",
                    label="stage manifest read_advice advised_bytes",
                ),
                "stage_read_advice_failures": 1 if raw_error is not None else 0,
                "stage_read_advice_error": raw_error,
            }
        )
    return telemetry


def _static_capacity_plan_from_compact_routes(
    *,
    batch_tokens: int,
    selected_compact_experts: tuple[int, ...],
    routes: list[dict[str, Any]],
    capacity_per_expert: int,
) -> StaticExpertCapacityPlan:
    capacity_per_expert = _positive_integer_value(
        capacity_per_expert,
        label="static_capacity_per_expert",
    )
    expert_to_assignments: dict[int, list[tuple[int, float]]] = {
        expert: [] for expert in selected_compact_experts
    }
    for route in routes:
        token_index = _nonnegative_integer_value(
            route.get("token_index"),
            label="compact route token_index",
        )
        if token_index >= batch_tokens:
            raise StagedMoEError("compact route token_index is outside batch range")
        experts = route.get("experts")
        weights = route.get("weights")
        if not isinstance(experts, list) or not isinstance(weights, list):
            raise StagedMoEError("compact route must include experts and weights")
        if len(experts) != len(weights):
            raise StagedMoEError("compact route experts/weights length mismatch")
        for raw_expert, raw_weight in zip(experts, weights):
            expert = _nonnegative_integer_value(
                raw_expert,
                label="compact route expert",
            )
            weight = _finite_float_value(raw_weight, label="compact route weight")
            if expert not in expert_to_assignments:
                raise StagedMoEError(f"compact route expert {expert} not in compact layout")
            expert_to_assignments[expert].append((token_index, weight))

    slots: list[StaticExpertTokenSlot] = []
    overflow: list[StaticExpertOverflow] = []
    usages: list[StaticExpertUsage] = []
    max_tokens_per_expert = 0
    for expert in selected_compact_experts:
        assignments = expert_to_assignments[expert]
        assigned = len(assignments)
        max_tokens_per_expert = max(max_tokens_per_expert, assigned)
        used = min(assigned, capacity_per_expert)
        usages.append(
            StaticExpertUsage(
                expert=expert,
                assigned_tokens=assigned,
                used_slots=used,
                overflow_assignments=assigned - used,
            )
        )
        for index, (token_index, weight) in enumerate(assignments):
            if index < capacity_per_expert:
                slots.append(
                    StaticExpertTokenSlot(
                        expert=expert,
                        slot=index,
                        token_index=token_index,
                        weight=weight,
                    )
                )
            else:
                overflow.append(
                    StaticExpertOverflow(
                        expert=expert,
                        overflow_index=index - capacity_per_expert,
                        token_index=token_index,
                        weight=weight,
                    )
                )

    total_capacity_slots = len(selected_compact_experts) * capacity_per_expert
    used_slots = len(slots)
    return StaticExpertCapacityPlan(
        batch_tokens=batch_tokens,
        selected_experts=selected_compact_experts,
        capacity_per_expert=capacity_per_expert,
        total_assignments=sum(len(route.get("experts", [])) for route in routes),
        total_capacity_slots=total_capacity_slots,
        used_slots=used_slots,
        utilization=(used_slots / total_capacity_slots if total_capacity_slots else 0.0),
        overflow_assignments=len(overflow),
        max_tokens_per_expert=max_tokens_per_expert,
        requires_overflow_path=bool(overflow),
        usages=tuple(usages),
        slots=tuple(slots),
        overflow=tuple(overflow),
    )


def _parse_stage_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_json(path)
    slots = manifest.get("slots")
    if not isinstance(slots, list) or not slots:
        raise StagedMoEError("stage manifest missing non-empty slots array")
    slot_items: list[dict[str, Any]] = []
    for item in slots:
        if not isinstance(item, dict):
            raise StagedMoEError("stage manifest slots must be objects")
        slot_items.append(item)
    return manifest, slot_items


def _validate_stage_file_size(
    manifest: dict[str, Any],
    *,
    stage_size: int,
) -> None:
    for field in ("planned_read_bytes", "staged_bytes"):
        if field in manifest:
            value = _nonnegative_integer_value(
                manifest.get(field),
                label=f"stage manifest {field}",
            )
            if value != stage_size:
                raise StagedMoEError(
                    f"stage manifest {field} {value} does not match stage file size "
                    f"{stage_size}"
                )


def _validate_stage_ranges(
    manifest: dict[str, Any],
    *,
    stage_size: int,
) -> list[dict[str, Any]]:
    ranges = manifest.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise StagedMoEError("stage manifest missing non-empty ranges array")

    parsed: list[dict[str, Any]] = []
    next_stage_offset = 0
    seen_indices: set[int] = set()
    for expected_index, item in enumerate(ranges):
        if not isinstance(item, dict):
            raise StagedMoEError("stage manifest ranges must be objects")
        range_index = _nonnegative_integer_value(
            item.get("range_index"),
            label="stage manifest range_index",
        )
        if range_index in seen_indices:
            raise StagedMoEError(f"stage manifest repeats range index {range_index}")
        seen_indices.add(range_index)
        if range_index != expected_index:
            raise StagedMoEError("stage manifest range indices must be contiguous")
        source_offset = _nonnegative_integer_value(
            item.get("source_offset"),
            label="stage manifest range source_offset",
        )
        source_length = _positive_integer_value(
            item.get("source_length"),
            label="stage manifest range source_length",
        )
        stage_offset = _nonnegative_integer_value(
            item.get("stage_offset"),
            label="stage manifest range stage_offset",
        )
        stage_length = _positive_integer_value(
            item.get("stage_length"),
            label="stage manifest range stage_length",
        )
        if stage_offset != next_stage_offset:
            raise StagedMoEError("stage manifest ranges must cover the stage file")
        if source_length != stage_length:
            raise StagedMoEError(
                "stage manifest range source_length must match stage_length"
            )
        next_stage_offset = stage_offset + stage_length
        if next_stage_offset > stage_size:
            raise StagedMoEError("stage manifest range is out of bounds")
        experts = item.get("experts")
        if not isinstance(experts, list) or not experts:
            raise StagedMoEError("stage manifest range missing experts")
        parsed.append(
            {
                "range_index": range_index,
                "source_offset": source_offset,
                "source_end": source_offset + source_length,
                "stage_offset": stage_offset,
                "stage_end": stage_offset + stage_length,
                "experts": tuple(
                    _nonnegative_integer_value(
                        expert,
                        label="stage manifest range expert",
                    )
                    for expert in experts
                ),
            }
        )
    if next_stage_offset != stage_size:
        raise StagedMoEError("stage manifest ranges must cover the stage file")
    return parsed


def _validate_stage_slots(
    *,
    slots: list[dict[str, Any]],
    selected_experts: tuple[int, ...],
    expert_physical_slots: tuple[int, ...],
    slot_bytes: int,
    stage_size: int,
    ranges: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    slot_by_expert: dict[int, dict[str, Any]] = {}
    range_experts: set[int] = set()
    for item in ranges:
        range_experts.update(item["experts"])
    selected_set = set(selected_experts)
    if range_experts != selected_set:
        raise StagedMoEError("stage manifest ranges do not match selected experts")

    for slot in slots:
        expert = _nonnegative_integer_value(
            slot.get("expert"),
            label="stage manifest slot expert",
        )
        if expert in slot_by_expert:
            raise StagedMoEError(f"stage manifest repeats slot for expert {expert}")
        source_offset = _nonnegative_integer_value(
            slot.get("source_offset"),
            label="stage manifest slot source_offset",
        )
        stage_offset = _nonnegative_integer_value(
            slot.get("stage_offset"),
            label="stage manifest slot stage_offset",
        )
        length = _positive_integer_value(
            slot.get("length"),
            label="stage manifest slot length",
        )
        if length != slot_bytes:
            raise StagedMoEError(
                f"stage slot for expert {expert} has length {length}, expected {slot_bytes}"
            )
        if expert_physical_slots:
            if expert >= len(expert_physical_slots):
                raise StagedMoEError(
                    f"stage slot for expert {expert} is outside expert_physical_slots"
                )
            physical_slot = expert_physical_slots[expert]
        else:
            physical_slot = expert
        expected_source_offset = physical_slot * slot_bytes
        if source_offset != expected_source_offset:
            raise StagedMoEError(
                f"stage slot for expert {expert} has source_offset {source_offset}, "
                f"expected {expected_source_offset}"
            )
        if stage_offset + length > stage_size:
            raise StagedMoEError(f"stage slot for expert {expert} is out of bounds")
        matching_range = next(
            (
                item
                for item in ranges
                if item["source_offset"] <= source_offset
                and source_offset + length <= item["source_end"]
                and expert in item["experts"]
            ),
            None,
        )
        if matching_range is None:
            raise StagedMoEError(
                f"stage slot for expert {expert} is not covered by a coalesced range"
            )
        expected_stage_offset = matching_range["stage_offset"] + (
            source_offset - matching_range["source_offset"]
        )
        if stage_offset != expected_stage_offset:
            raise StagedMoEError(
                f"stage slot for expert {expert} has stage_offset {stage_offset}, "
                f"expected {expected_stage_offset}"
            )
        slot_by_expert[expert] = slot

    if set(slot_by_expert) != selected_set:
        raise StagedMoEError("stage manifest slots do not match selected experts")
    return slot_by_expert


def _stage_manifest_expert_physical_slots(manifest: dict[str, Any]) -> tuple[int, ...]:
    batch_plan = manifest.get("batch_plan")
    if not isinstance(batch_plan, dict):
        return ()
    io_plan = batch_plan.get("io_plan")
    if not isinstance(io_plan, dict):
        return ()
    raw_slots = io_plan.get("expert_physical_slots")
    if raw_slots in (None, []):
        return ()
    if not isinstance(raw_slots, list):
        raise StagedMoEError("stage manifest expert_physical_slots must be an array")
    slots = tuple(
        _nonnegative_integer_value(
            slot,
            label="stage manifest expert_physical_slots entry",
        )
        for slot in raw_slots
    )
    if sorted(slots) != list(range(len(slots))):
        raise StagedMoEError("stage manifest expert_physical_slots must be a permutation")
    return slots


def _stage_file_is_compact(
    *,
    selected_experts: tuple[int, ...],
    slot_by_expert: dict[int, dict[str, Any]],
    slot_bytes: int,
    stage_size: int,
) -> bool:
    if stage_size != len(selected_experts) * slot_bytes:
        return False
    for compact_index, expert in enumerate(selected_experts):
        slot = slot_by_expert.get(expert)
        if slot is None:
            return False
        stage_offset = _nonnegative_integer_value(
            slot.get("stage_offset"),
            label="stage manifest slot stage_offset",
        )
        length = _positive_integer_value(
            slot.get("length"),
            label="stage manifest slot length",
        )
        if length != slot_bytes or stage_offset != compact_index * slot_bytes:
            return False
    return True


def _try_hardlink_compact_stage(*, source: Path, destination: Path) -> bool:
    if source.resolve(strict=False) == destination.resolve(strict=False):
        return False
    _remove_partial_file(destination)
    try:
        os.link(source, destination)
    except OSError:
        _remove_partial_file(destination)
        return False
    return True


def _build_compact_stage(
    *,
    manifest: dict[str, Any],
    slots: list[dict[str, Any]],
    output_dir: Path,
    max_compact_stage_bytes: int,
    copy_chunk_bytes: int,
    disk_safety_margin_bytes: int,
) -> tuple[Path, Path, tuple[int, ...], int, int, str]:
    expert_layout_raw = manifest.get("expert_layout_path")
    stage_file_raw = manifest.get("stage_file_path")
    if not isinstance(expert_layout_raw, str) or not expert_layout_raw:
        raise StagedMoEError("stage manifest missing expert_layout_path")
    if not isinstance(stage_file_raw, str) or not stage_file_raw:
        raise StagedMoEError("stage manifest missing stage_file_path")
    expert_layout_path = Path(expert_layout_raw)
    stage_file_path = Path(stage_file_raw)
    layer = _nonnegative_integer_value(manifest.get("layer"), label="stage manifest layer")
    slot_bytes = _positive_integer_value(
        manifest.get("expert_slot_bytes"),
        label="stage manifest expert_slot_bytes",
    )
    selected_experts_raw = manifest.get("selected_experts")
    if not isinstance(selected_experts_raw, list) or not selected_experts_raw:
        raise StagedMoEError("stage manifest is missing selected experts or slot bytes")
    selected_experts = tuple(
        _nonnegative_integer_value(
            expert,
            label="stage manifest selected_experts",
        )
        for expert in selected_experts_raw
    )
    if len(set(selected_experts)) != len(selected_experts):
        raise StagedMoEError("stage manifest selected_experts must be unique")
    if selected_experts != tuple(sorted(selected_experts)):
        raise StagedMoEError("stage manifest selected_experts must be sorted")
    compact_bytes = len(selected_experts) * slot_bytes
    if compact_bytes > max_compact_stage_bytes:
        raise StagedMoEError(
            f"compact stage bytes {compact_bytes} exceed limit {max_compact_stage_bytes}"
        )
    try:
        stage_size = stage_file_path.stat().st_size
    except OSError as exc:
        raise StagedMoEError(f"failed to stat stage file {stage_file_path}: {exc}") from exc
    _validate_stage_file_size(manifest, stage_size=stage_size)
    ranges = _validate_stage_ranges(manifest, stage_size=stage_size)
    slot_by_expert = _validate_stage_slots(
        slots=slots,
        selected_experts=selected_experts,
        expert_physical_slots=_stage_manifest_expert_physical_slots(manifest),
        slot_bytes=slot_bytes,
        stage_size=stage_size,
        ranges=ranges,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    compact_layer_path = output_dir / "compact_stage.bin"
    compact_layout_path = output_dir / "compact_layout.json"
    compact_storage = "copy"
    compact_materialized_bytes = compact_bytes
    if _stage_file_is_compact(
        selected_experts=selected_experts,
        slot_by_expert=slot_by_expert,
        slot_bytes=slot_bytes,
        stage_size=stage_size,
    ) and _try_hardlink_compact_stage(
        source=stage_file_path,
        destination=compact_layer_path,
    ):
        compact_storage = "hardlink"
        compact_materialized_bytes = 0
    else:
        budget = disk_budget(
            output_dir,
            compact_bytes,
            safety_margin_bytes=disk_safety_margin_bytes,
        )
        if not budget.ok:
            raise StagedMoEError(
                "not enough free disk for compact stage file: "
                f"need {compact_bytes + disk_safety_margin_bytes} bytes including margin, "
                f"have {budget.available_bytes} bytes"
            )
        try:
            copy_buffer = bytearray(copy_chunk_bytes)
            with stage_file_path.open("rb") as source, compact_layer_path.open("wb") as dest:
                for expert in selected_experts:
                    slot = slot_by_expert.get(expert)
                    if slot is None:
                        raise StagedMoEError(
                            f"stage manifest missing slot for expert {expert}"
                        )
                    stage_offset = _nonnegative_integer_value(
                        slot.get("stage_offset"),
                        label="stage manifest slot stage_offset",
                    )
                    _copy_exact_range(
                        source=source,
                        destination=dest,
                        source_offset=stage_offset,
                        length=slot_bytes,
                        copy_chunk_bytes=copy_chunk_bytes,
                        copy_buffer=copy_buffer,
                    )
        except (OSError, StagedMoEError) as exc:
            _remove_partial_file(compact_layer_path)
            raise StagedMoEError(f"failed to build compact stage file: {exc}") from exc

    original_layout = _load_json(expert_layout_path)
    original_layer = _find_layer(original_layout, layer)
    components = original_layer.get("components")
    if not isinstance(components, list):
        raise StagedMoEError("original expert layout missing components array")
    compact_layout = {
        "version": original_layout.get("version", 1),
        "model_type": original_layout.get("model_type", "glm_moe_dsa"),
        "config_sha256": original_layout.get("config_sha256"),
        "quantization": original_layout.get("quantization"),
        "group_size": original_layout.get("group_size"),
        "num_layers": 1,
        "num_experts": len(selected_experts),
        "component_order": original_layout.get("component_order", []),
        "layers": [
            {
                "layer": layer,
                "num_experts": len(selected_experts),
                "expert_slot_bytes": slot_bytes,
                "layer_file": compact_layer_path.name,
                "components": components,
            }
        ],
        "original_experts": list(selected_experts),
    }
    try:
        _write_json_atomic(compact_layout_path, compact_layout)
    except OSError as exc:
        _remove_partial_file(compact_layer_path)
        raise StagedMoEError(f"failed to write compact layout {compact_layout_path}: {exc}") from exc
    return (
        compact_layout_path,
        compact_layer_path,
        selected_experts,
        compact_bytes,
        compact_materialized_bytes,
        compact_storage,
    )


def run_staged_routed_moe_batch(
    *,
    runner_path: str | Path,
    stage_manifest_path: str | Path,
    input_f32_path: str | Path,
    output_f32_path: str | Path,
    output_dir: str | Path,
    max_compact_stage_mib: float = 4096.0,
    copy_chunk_mib: float = 8.0,
    disk_safety_margin_bytes: int = 0,
    max_slot_mib: int = 256,
    max_runner_scratch_mib: int = 4096,
    moe_token_block: MoETokenBlock = "auto",
    static_capacity_per_expert: int | None = None,
    static_capacity_output_json_path: str | Path | None = None,
    static_capacity_output_bin_path: str | Path | None = None,
    write_static_capacity_json: bool = True,
    allow_static_capacity_overflow: bool = False,
    keep_token_files: bool = False,
    echo_runner_output: bool = True,
    moe_plan_server_session: StagedRoutedMoEBatchPlanServerSession | None = None,
    moe_output_accumulator: MoEOutputAccumulator = "env",
) -> StagedRoutedMoEBatchResult:
    wall_started = time.perf_counter()
    max_compact_stage_mib = _positive_float_value(
        max_compact_stage_mib,
        label="max_compact_stage_mib",
    )
    copy_chunk_mib = _positive_float_value(copy_chunk_mib, label="copy_chunk_mib")
    max_slot_mib = _positive_integer_value(max_slot_mib, label="max_slot_mib")
    max_runner_scratch_mib = _positive_integer_value(
        max_runner_scratch_mib,
        label="max_runner_scratch_mib",
    )
    moe_token_block_arg, moe_token_block_value = _normalize_moe_token_block(moe_token_block)
    moe_output_accumulator = _normalize_moe_output_accumulator(
        moe_output_accumulator
    )
    if static_capacity_per_expert is not None:
        static_capacity_per_expert = _positive_integer_value(
            static_capacity_per_expert,
            label="static_capacity_per_expert",
        )
    disk_safety_margin_bytes = _nonnegative_integer_value(
        disk_safety_margin_bytes,
        label="disk_safety_margin_bytes",
    )
    copy_chunk_bytes = max(1, int(copy_chunk_mib * 1024 * 1024))
    max_compact_stage_bytes = int(max_compact_stage_mib * 1024 * 1024)
    manifest_path = Path(stage_manifest_path)
    manifest, slots = _parse_stage_manifest(manifest_path)
    stage_telemetry = _stage_manifest_telemetry(manifest)
    out_dir = Path(output_dir)
    (
        compact_layout,
        compact_layer,
        selected_experts,
        compact_stage_bytes,
        compact_stage_materialized_bytes,
        compact_stage_storage,
    ) = (None, None, (), 0, 0, "")
    compact_started = time.perf_counter()
    try:
        (
            compact_layout,
            compact_layer,
            selected_experts,
            compact_stage_bytes,
            compact_stage_materialized_bytes,
            compact_stage_storage,
        ) = _build_compact_stage(
            manifest=manifest,
            slots=slots,
            output_dir=out_dir,
            max_compact_stage_bytes=max_compact_stage_bytes,
            copy_chunk_bytes=copy_chunk_bytes,
            disk_safety_margin_bytes=disk_safety_margin_bytes,
        )
    finally:
        wall_compact_stage_elapsed = time.perf_counter() - compact_started
    layer = _nonnegative_integer_value(manifest.get("layer"), label="stage manifest layer")
    batch_plan = manifest.get("batch_plan")
    if not isinstance(batch_plan, dict):
        raise StagedMoEError("stage manifest missing batch_plan")
    token_routes = batch_plan.get("token_routes")
    if not isinstance(token_routes, list) or not token_routes:
        raise StagedMoEError("stage manifest batch_plan missing token_routes")
    compact_expert = {expert: index for index, expert in enumerate(selected_experts)}
    expert_layout_raw = manifest.get("expert_layout_path")
    if not isinstance(expert_layout_raw, str) or not expert_layout_raw:
        raise StagedMoEError("stage manifest missing expert_layout_path")
    original_layout = _load_json(Path(expert_layout_raw))
    hidden_dim, _packed_in = _component_shape2(
        _find_layer(original_layout, layer), "down_proj.weight"
    )
    token_bytes = hidden_dim * 4
    input_path = Path(input_f32_path)
    output_path = Path(output_f32_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise StagedMoEError(f"failed to stat staged MoE input {input_path}: {exc}") from exc
    expected_input_bytes = len(token_routes) * token_bytes
    if input_bytes != expected_input_bytes:
        raise StagedMoEError(
            f"input bytes {input_bytes} do not match token_routes*hidden_dim*f32 "
            f"({expected_input_bytes})"
        )
    runner = Path(runner_path)
    if not runner.exists():
        raise StagedMoEError(f"runner not found: {runner}")
    token_dir = out_dir / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    compact_routes_path = out_dir / "compact_routes.json"
    routes: list[dict[str, Any]] = []
    max_route_k = 0
    routes_started = time.perf_counter()
    try:
        seen_token_indices: set[int] = set()
        for route in token_routes:
            if not isinstance(route, dict):
                raise StagedMoEError("token route entries must be objects")
            token_index = _nonnegative_integer_value(
                route.get("token_index"),
                label="token route token_index",
            )
            if token_index >= len(token_routes):
                raise StagedMoEError("token route token_index is outside batch range")
            if token_index in seen_token_indices:
                raise StagedMoEError("token route token_index values must be unique")
            seen_token_indices.add(token_index)
            experts = route.get("experts")
            weights = route.get("weights")
            if not isinstance(experts, list) or not isinstance(weights, list):
                raise StagedMoEError("token route must include experts and weights")
            if len(experts) != len(weights) or not experts:
                raise StagedMoEError("token route experts/weights are invalid")
            compact_ids: list[int] = []
            weight_items: list[float] = []
            for raw_expert, raw_weight in zip(experts, weights):
                expert = _nonnegative_integer_value(
                    raw_expert,
                    label="token route expert",
                )
                if expert not in compact_expert:
                    raise StagedMoEError(
                        "token route contains expert not present in compact stage"
                    )
                compact_ids.append(compact_expert[expert])
                weight_items.append(
                    _finite_float_value(raw_weight, label="token route weight")
                )
            max_route_k = max(max_route_k, len(compact_ids))
            routes.append(
                {
                    "token_index": token_index,
                    "experts": compact_ids,
                    "weights": weight_items,
                }
            )
        _write_json_atomic(
            compact_routes_path,
            {
                "version": 1,
                "batch_tokens": len(routes),
                "max_k": max_route_k,
                "routes": routes,
            },
        )
    except StagedMoEError:
        _remove_partial_file(compact_routes_path)
        _remove_partial_file(compact_layout)
        _remove_partial_file(compact_layer)
        raise
    except OSError as exc:
        _remove_partial_file(compact_routes_path)
        _remove_partial_file(compact_layout)
        _remove_partial_file(compact_layer)
        raise StagedMoEError(f"failed to write compact routes {compact_routes_path}: {exc}") from exc
    finally:
        wall_routes_elapsed = time.perf_counter() - routes_started

    static_capacity_path: Path | None = None
    static_capacity_binary_path: Path | None = None
    static_capacity_used_slots = 0
    static_capacity_total_slots = 0
    static_capacity_overflow = 0
    static_capacity_binary_bytes = 0
    wall_static_capacity_elapsed: float | None = None
    if static_capacity_per_expert is not None:
        static_capacity_started = time.perf_counter()
        static_capacity_plan = _static_capacity_plan_from_compact_routes(
            batch_tokens=len(routes),
            selected_compact_experts=tuple(range(len(selected_experts))),
            routes=routes,
            capacity_per_expert=static_capacity_per_expert,
        )
        static_capacity_path = (
            Path(static_capacity_output_json_path)
            if static_capacity_output_json_path is not None
            else (
                out_dir / "static_capacity.json"
                if write_static_capacity_json
                else None
            )
        )
        static_capacity_binary_path = (
            Path(static_capacity_output_bin_path)
            if static_capacity_output_bin_path is not None
            else out_dir / "static_capacity.bin"
        )
        static_capacity_json_bytes = (
            static_expert_capacity_json_bytes(static_capacity_plan)
            if static_capacity_path is not None
            else 0
        )
        _require_static_capacity_disk_budget(
            json_path=static_capacity_path,
            json_bytes=static_capacity_json_bytes,
            binary_path=static_capacity_binary_path,
            binary_bytes=static_expert_capacity_binary_bytes(static_capacity_plan),
            safety_margin_bytes=disk_safety_margin_bytes,
        )
        try:
            if static_capacity_path is not None:
                write_static_expert_capacity_plan(
                    static_capacity_plan,
                    static_capacity_path,
                    allow_overflow=allow_static_capacity_overflow,
                )
            binary_report = write_static_expert_capacity_binary(
                static_capacity_plan,
                static_capacity_binary_path,
                allow_overflow=allow_static_capacity_overflow,
            )
            binary_validation = validate_static_expert_capacity_binary(
                static_capacity_binary_path,
                expected_plan=static_capacity_plan,
            )
            if binary_validation.bytes_read != binary_report.bytes_written:
                raise ExpertIOPlanError(
                    "static capacity binary validation byte mismatch"
                )
        except ExpertIOPlanError as exc:
            if static_capacity_path is not None:
                _remove_partial_file(static_capacity_path)
            _remove_partial_file(static_capacity_binary_path)
            raise StagedMoEError(f"failed to write static capacity plan: {exc}") from exc
        static_capacity_used_slots = static_capacity_plan.used_slots
        static_capacity_total_slots = static_capacity_plan.total_capacity_slots
        static_capacity_overflow = static_capacity_plan.overflow_assignments
        static_capacity_binary_bytes = binary_validation.bytes_read
        wall_static_capacity_elapsed = time.perf_counter() - static_capacity_started

    routes_flag = "--routes-bin" if static_capacity_binary_path is not None else "--routes-json"
    routes_path = static_capacity_binary_path if static_capacity_binary_path is not None else compact_routes_path
    cmd = [
        str(runner),
        "--layout",
        str(compact_layout),
        "--layer",
        str(layer),
        "--run-moe-batch",
        routes_flag,
        str(routes_path),
        "--input-f32",
        str(input_path),
        "--batch-tokens",
        str(len(routes)),
        "--output-f32",
        str(output_path),
        "--max-k",
        str(max_route_k),
        "--max-slot-mib",
        str(max_slot_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
        "--moe-token-block",
        moe_token_block_arg,
    ]
    runner_started = time.perf_counter()
    if moe_plan_server_session is None:
        runner_stdout = _run_command(
            cmd,
            echo_output=echo_runner_output,
            env=_moe_output_accumulator_env(moe_output_accumulator),
        )
        command_count = 1
        first_command = tuple(cmd)
    else:
        plan_path = out_dir / "moe_batch_plan.json"
        write_staged_routed_moe_batch_plan(
            plan_path,
            [
                StagedRoutedMoEBatchPlanJob(
                    layout_path=compact_layout,
                    layer=layer,
                    routes_json_path=(
                        compact_routes_path
                        if static_capacity_binary_path is None
                        else None
                    ),
                    routes_bin_path=static_capacity_binary_path,
                    input_path=input_path,
                    output_path=output_path,
                    batch_tokens=len(routes),
                    max_k=max_route_k,
                    max_slot_mib=max_slot_mib,
                    max_runner_scratch_mib=max_runner_scratch_mib,
                    moe_token_block=moe_token_block_value,
                )
            ],
        )
        plan_result = moe_plan_server_session.submit_plan(plan_path)
        runner_stdout = plan_result.runner_stdout or ""
        command_count = 0
        first_command = plan_result.first_command
    wall_runner_elapsed = time.perf_counter() - runner_started
    output_validation_started = time.perf_counter()
    try:
        output_bytes = output_path.stat().st_size
    except OSError as exc:
        raise StagedMoEError(f"failed to stat staged MoE output {output_path}: {exc}") from exc
    if output_bytes != expected_input_bytes:
        raise StagedMoEError(
            f"output bytes {output_bytes} do not match expected {expected_input_bytes}"
        )
    wall_output_validation_elapsed = time.perf_counter() - output_validation_started
    wall_total_elapsed = time.perf_counter() - wall_started
    return StagedRoutedMoEBatchResult(
        runner_path=runner,
        stage_manifest_path=manifest_path,
        stage_file_path=Path(str(manifest.get("stage_file_path"))),
        compact_layout_path=compact_layout,
        compact_layer_path=compact_layer,
        compact_routes_path=compact_routes_path,
        static_capacity_path=static_capacity_path,
        static_capacity_binary_path=static_capacity_binary_path,
        input_path=input_path,
        output_path=output_path,
        output_dir=out_dir,
        layer=layer,
        batch_tokens=len(token_routes),
        hidden_dim=hidden_dim,
        selected_experts=selected_experts,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        token_bytes=token_bytes,
        compact_stage_bytes=compact_stage_bytes,
        compact_stage_materialized_bytes=compact_stage_materialized_bytes,
        compact_stage_storage=compact_stage_storage,
        max_compact_stage_bytes=max_compact_stage_bytes,
        stage_io_summary_available=bool(
            stage_telemetry["stage_io_summary_available"]
        ),
        stage_read_advice_available=bool(
            stage_telemetry["stage_read_advice_available"]
        ),
        stage_serial_read_bytes=int(stage_telemetry["stage_serial_read_bytes"]),
        stage_unique_requested_bytes=int(
            stage_telemetry["stage_unique_requested_bytes"]
        ),
        stage_planned_read_bytes=int(stage_telemetry["stage_planned_read_bytes"]),
        stage_staged_bytes=int(stage_telemetry["stage_staged_bytes"]),
        stage_waste_bytes=int(stage_telemetry["stage_waste_bytes"]),
        stage_coalesced_savings_bytes=int(
            stage_telemetry["stage_coalesced_savings_bytes"]
        ),
        stage_raw_range_count=int(stage_telemetry["stage_raw_range_count"]),
        stage_coalesced_range_count=int(
            stage_telemetry["stage_coalesced_range_count"]
        ),
        stage_assignment_read_amplification=float(
            stage_telemetry["stage_assignment_read_amplification"]
        ),
        stage_unique_read_amplification=float(
            stage_telemetry["stage_unique_read_amplification"]
        ),
        stage_staged_unique_read_amplification=float(
            stage_telemetry["stage_staged_unique_read_amplification"]
        ),
        stage_budget_utilization=float(stage_telemetry["stage_budget_utilization"]),
        stage_read_advice_supported=(
            stage_telemetry["stage_read_advice_supported"]
            if stage_telemetry["stage_read_advice_supported"] is None
            else bool(stage_telemetry["stage_read_advice_supported"])
        ),
        stage_read_advice_attempted_ranges=int(
            stage_telemetry["stage_read_advice_attempted_ranges"]
        ),
        stage_read_advice_calls=int(stage_telemetry["stage_read_advice_calls"]),
        stage_read_advice_bytes=int(stage_telemetry["stage_read_advice_bytes"]),
        stage_read_advice_failures=int(
            stage_telemetry["stage_read_advice_failures"]
        ),
        stage_read_advice_error=(
            stage_telemetry["stage_read_advice_error"]
            if stage_telemetry["stage_read_advice_error"] is None
            else str(stage_telemetry["stage_read_advice_error"])
        ),
        copy_chunk_bytes=copy_chunk_bytes,
        stage_copy_seconds_ok=(
            stage_telemetry["stage_copy_seconds_ok"]
            if stage_telemetry["stage_copy_seconds_ok"] is None
            else bool(stage_telemetry["stage_copy_seconds_ok"])
        ),
        static_capacity_per_expert=static_capacity_per_expert,
        static_capacity_used_slots=static_capacity_used_slots,
        static_capacity_total_slots=static_capacity_total_slots,
        static_capacity_overflow_assignments=static_capacity_overflow,
        static_capacity_binary_bytes=static_capacity_binary_bytes,
        moe_token_block=moe_token_block_value,
        moe_token_block_mode=_runner_stat(runner_stdout, "token block mode"),
        effective_moe_token_block=_runner_stat_int(runner_stdout, "token block"),
        moe_max_expert_tokens=_runner_stat_int(runner_stdout, "max expert tokens"),
        moe_batch_buffer_bytes=_runner_stat_int(runner_stdout, "batch buffer bytes"),
        moe_estimated_peak_bytes=_runner_stat_int(runner_stdout, "estimated peak"),
        moe_mxfp4_token_tile=_runner_stat_int(runner_stdout, "MXFP4 token tile"),
        moe_mxfp4_vector_swiglu=_runner_stat_bool(
            runner_stdout, "MXFP4 vector SwiGLU"
        ),
        moe_mxfp4_group32_specialized=_runner_stat_bool(
            runner_stdout, "MXFP4 group32 path"
        ),
        moe_mxfp4_swiglu_activation=_runner_stat(runner_stdout, "MXFP4 SwiGLU act"),
        moe_output_accumulator=_runner_stat(runner_stdout, "output accumulator"),
        moe_output_accumulator_bytes=_runner_stat_int(
            runner_stdout, "output accum bytes"
        ),
        command_count=command_count,
        first_command=first_command,
        stage_copy_elapsed_seconds=(
            None
            if stage_telemetry["stage_copy_elapsed_seconds"] is None
            else float(stage_telemetry["stage_copy_elapsed_seconds"])
        ),
        stage_copy_throughput_gib_per_second=(
            None
            if stage_telemetry["stage_copy_throughput_gib_per_second"] is None
            else float(stage_telemetry["stage_copy_throughput_gib_per_second"])
        ),
        moe_timing_sort_seconds=_runner_stat_float(runner_stdout, "timing sort"),
        moe_timing_setup_seconds=_runner_stat_float(runner_stdout, "timing setup"),
        moe_timing_expert_read_seconds=_runner_stat_float(
            runner_stdout, "timing expert read"
        ),
        moe_timing_input_read_seconds=_runner_stat_float(
            runner_stdout, "timing input read"
        ),
        moe_timing_output_read_seconds=_runner_stat_float(
            runner_stdout, "timing output read"
        ),
        moe_timing_kernel_seconds=_runner_stat_float(runner_stdout, "timing kernel"),
        moe_timing_mxfp4_swiglu_kernel_seconds=_runner_stat_float(
            runner_stdout, "timing mxfp4 swiglu kernel"
        ),
        moe_timing_mxfp4_down_add_kernel_seconds=_runner_stat_float(
            runner_stdout, "timing mxfp4 down add kernel"
        ),
        moe_timing_output_write_seconds=_runner_stat_float(
            runner_stdout, "timing output write"
        ),
        moe_timing_final_read_seconds=_runner_stat_float(
            runner_stdout, "timing final read"
        ),
        moe_timing_total_seconds=_runner_stat_float(runner_stdout, "timing total"),
        wall_total_elapsed_seconds=wall_total_elapsed,
        wall_compact_stage_elapsed_seconds=wall_compact_stage_elapsed,
        wall_routes_elapsed_seconds=wall_routes_elapsed,
        wall_static_capacity_elapsed_seconds=wall_static_capacity_elapsed,
        wall_runner_elapsed_seconds=wall_runner_elapsed,
        wall_output_validation_elapsed_seconds=wall_output_validation_elapsed,
    )


def _router_token_routes_payload(token_routes: object) -> list[dict[str, Any]]:
    if not isinstance(token_routes, (list, tuple)):
        raise StagedMoEError("token_routes must be a sequence")
    payload: list[dict[str, Any]] = []
    for route in token_routes:
        token_index = getattr(route, "token_index", None)
        experts = getattr(route, "experts", None)
        weights = getattr(route, "weights", None)
        if token_index is None and isinstance(route, dict):
            token_index = route.get("token_index")
            experts = route.get("experts")
            weights = route.get("weights")
        payload.append(
            {
                "token_index": _nonnegative_integer_value(
                    token_index,
                    label="token route token_index",
                ),
                "experts": list(experts) if isinstance(experts, (list, tuple)) else [],
                "weights": list(weights) if isinstance(weights, (list, tuple)) else [],
            }
        )
    return payload


def run_tiled_staged_routed_moe_batch(
    *,
    runner_path: str | Path,
    expert_layout_path: str | Path,
    layer: int,
    router_json_dir: str | Path,
    input_f32_path: str | Path,
    output_f32_path: str | Path,
    output_dir: str | Path,
    router_json_glob: str = "*.router.json",
    merge_gap_bytes: int = 0,
    align_bytes: int = 4096,
    max_stage_mib: float = 4096.0,
    max_compact_stage_mib: float = 4096.0,
    copy_chunk_mib: float = 8.0,
    disk_safety_margin_bytes: int = 0,
    ssd_read_gib_per_second: float | int = 0.0,
    max_read_seconds: float | int = 0.0,
    max_raw_ranges: int = 0,
    max_coalesced_ranges: int = 0,
    max_slot_mib: int = 256,
    max_runner_scratch_mib: int = 4096,
    moe_token_block: MoETokenBlock = "auto",
    static_capacity_per_expert: int | None = None,
    write_static_capacity_json: bool = True,
    allow_static_capacity_overflow: bool = False,
    keep_token_files: bool = False,
    echo_runner_output: bool = True,
    moe_plan_server_session: StagedRoutedMoEBatchPlanServerSession | None = None,
    moe_output_accumulator: MoEOutputAccumulator = "env",
) -> TiledStagedRoutedMoEBatchResult:
    wall_started = time.perf_counter()
    layer = _nonnegative_integer_value(layer, label="layer")
    moe_output_accumulator = _normalize_moe_output_accumulator(
        moe_output_accumulator
    )
    max_read_seconds = _finite_float_value(max_read_seconds, label="max_read_seconds")
    if max_read_seconds < 0:
        raise StagedMoEError("max_read_seconds must be non-negative")
    plan_started = time.perf_counter()
    tiling = plan_batch_expert_io_tiles(
        expert_layout_path,
        layer=layer,
        router_json_dir=router_json_dir,
        router_json_glob=router_json_glob,
        merge_gap_bytes=merge_gap_bytes,
        align_bytes=align_bytes,
        max_stage_mib=max_stage_mib,
        max_compact_stage_mib=max_compact_stage_mib,
        ssd_read_gib_per_second=ssd_read_gib_per_second,
    )
    wall_plan_elapsed = time.perf_counter() - plan_started
    batch_tokens = tiling.batch_plan.batch_tokens
    input_path = Path(input_f32_path)
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise StagedMoEError(f"failed to stat tiled MoE input {input_path}: {exc}") from exc
    if input_bytes == 0 or input_bytes % (batch_tokens * 4) != 0:
        raise StagedMoEError("tiled MoE input bytes do not divide batch_tokens*f32")
    hidden_dim = input_bytes // (batch_tokens * 4)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_f32_path)
    route_payloads = _router_token_routes_payload(tiling.batch_plan.token_routes)
    tile_results: list[StagedRoutedMoEBatchResult] = []
    tile_stage_results: list[BatchExpertStageResult] = []
    tile_original_indices: list[tuple[int, ...]] = []
    elapsed_read_seconds = 0.0
    wall_tile_router_elapsed = 0.0
    wall_tile_input_elapsed = 0.0
    wall_stage_elapsed = 0.0
    wall_staged_moe_elapsed = 0.0
    wall_scatter_elapsed = 0.0
    wall_cleanup_elapsed = 0.0
    try:
        for tile in tiling.tiles:
            tile_dir = out_dir / f"tile_{tile.tile_index:04d}"
            tile_router_dir = tile_dir / "router_json"
            tile_router_started = time.perf_counter()
            original_indices, _router_dir = _write_tile_router_jsons(
                output_dir=tile_router_dir,
                token_routes=route_payloads,
                selected_experts=tile.selected_experts,
            )
            wall_tile_router_elapsed += time.perf_counter() - tile_router_started
            tile_original_indices.append(original_indices)
            tile_input = tile_dir / "input.f32"
            tile_input_started = time.perf_counter()
            _gather_f32_rows(
                input_path=input_path,
                output_path=tile_input,
                token_indices=original_indices,
                batch_tokens=batch_tokens,
                hidden_dim=hidden_dim,
            )
            wall_tile_input_elapsed += time.perf_counter() - tile_input_started
            stage_read_seconds_budget = 0.0
            if max_read_seconds > 0:
                stage_read_seconds_budget = max_read_seconds - elapsed_read_seconds
                if stage_read_seconds_budget <= 0:
                    raise StagedMoEError("no read-time budget remains for next tile")
            stage_started = time.perf_counter()
            stage_result = stage_batch_experts(
                expert_layout_path,
                layer=layer,
                router_json_dir=tile_router_dir,
                stage_file_path=tile_dir / "experts.stage.bin",
                manifest_path=tile_dir / "experts.stage.manifest.json",
                router_json_glob=router_json_glob,
                merge_gap_bytes=merge_gap_bytes,
                align_bytes=align_bytes,
                max_stage_mib=max_stage_mib,
                copy_chunk_mib=copy_chunk_mib,
                disk_safety_margin_bytes=disk_safety_margin_bytes,
                ssd_read_gib_per_second=ssd_read_gib_per_second,
                max_read_seconds=stage_read_seconds_budget,
                max_raw_ranges=max_raw_ranges,
                max_coalesced_ranges=max_coalesced_ranges,
            )
            wall_stage_elapsed += time.perf_counter() - stage_started
            tile_stage_results.append(stage_result)
            if stage_result.io_summary.planned_read_seconds is not None:
                elapsed_read_seconds += stage_result.io_summary.planned_read_seconds
            tile_output = tile_dir / "output.f32"
            staged_moe_started = time.perf_counter()
            result = run_staged_routed_moe_batch(
                runner_path=runner_path,
                stage_manifest_path=stage_result.manifest_path,
                input_f32_path=tile_input,
                output_f32_path=tile_output,
                output_dir=tile_dir / "staged_moe",
                max_compact_stage_mib=max_compact_stage_mib,
                copy_chunk_mib=copy_chunk_mib,
                disk_safety_margin_bytes=disk_safety_margin_bytes,
                max_slot_mib=max_slot_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                moe_token_block=moe_token_block,
                static_capacity_per_expert=static_capacity_per_expert,
                write_static_capacity_json=write_static_capacity_json,
                allow_static_capacity_overflow=allow_static_capacity_overflow,
                keep_token_files=keep_token_files,
                echo_runner_output=echo_runner_output,
                moe_plan_server_session=moe_plan_server_session,
                moe_output_accumulator=moe_output_accumulator,
            )
            wall_staged_moe_elapsed += time.perf_counter() - staged_moe_started
            tile_results.append(result)
            scatter_started = time.perf_counter()
            _scatter_add_f32_rows(
                accumulator_path=output_path,
                partial_path=tile_output,
                token_indices=original_indices,
                batch_tokens=batch_tokens,
                hidden_dim=hidden_dim,
                initialize=tile.tile_index == 0,
            )
            wall_scatter_elapsed += time.perf_counter() - scatter_started
            if not keep_token_files:
                cleanup_started = time.perf_counter()
                shutil.rmtree(tile_router_dir, ignore_errors=True)
                wall_cleanup_elapsed += time.perf_counter() - cleanup_started
    except Exception:
        _remove_partial_file(output_path)
        raise
    output_validation_started = time.perf_counter()
    try:
        output_bytes = output_path.stat().st_size
    except OSError as exc:
        raise StagedMoEError(f"failed to stat tiled MoE output {output_path}: {exc}") from exc
    wall_output_validation_elapsed = time.perf_counter() - output_validation_started
    wall_total_elapsed = time.perf_counter() - wall_started
    return TiledStagedRoutedMoEBatchResult(
        runner_path=Path(runner_path),
        expert_layout_path=Path(expert_layout_path),
        input_path=input_path,
        output_path=output_path,
        output_dir=out_dir,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        tile_count=len(tile_results),
        tile_original_token_indices=tuple(tile_original_indices),
        tile_stage_results=tuple(tile_stage_results),
        tile_results=tuple(tile_results),
        total_stage_planned_read_bytes=sum(
            item.stage_planned_read_bytes for item in tile_results
        ),
        max_tile_stage_planned_read_bytes=max(
            (item.stage_planned_read_bytes for item in tile_results),
            default=0,
        ),
        total_compact_stage_bytes=sum(item.compact_stage_bytes for item in tile_results),
        max_tile_compact_stage_bytes=max(
            (item.compact_stage_bytes for item in tile_results),
            default=0,
        ),
        output_bytes=output_bytes,
        wall_total_elapsed_seconds=wall_total_elapsed,
        wall_plan_elapsed_seconds=wall_plan_elapsed,
        wall_tile_router_elapsed_seconds=wall_tile_router_elapsed,
        wall_tile_input_elapsed_seconds=wall_tile_input_elapsed,
        wall_stage_elapsed_seconds=wall_stage_elapsed,
        wall_staged_moe_elapsed_seconds=wall_staged_moe_elapsed,
        wall_scatter_elapsed_seconds=wall_scatter_elapsed,
        wall_cleanup_elapsed_seconds=wall_cleanup_elapsed,
        wall_output_validation_elapsed_seconds=wall_output_validation_elapsed,
    )
