from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from array import array
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .decode_cache import load_decode_cache_layout
from .dsa_indexer import (
    DSAIndexerBatchResult,
    DSAIndexerError,
    run_dsa_indexer_batch,
    write_dsa_index_cache_batch,
)
from .expert_io import BatchExpertStageResult, ExpertIOPlanError, stage_batch_experts
from .runtime_check import RuntimeCheckError, check_layer_runtime
from .resident_affine import (
    ResidentAffineLayoutError,
    resident_affine_int4_layout_info,
    resident_mxfp4_layout_info,
)
from .staged_moe import (
    MoEOutputAccumulator,
    MoETokenBlock,
    StagedMoEError,
    StagedRoutedMoEBatchPlanServerSession,
    StagedRoutedMoEBatchResult,
    TiledStagedRoutedMoEBatchResult,
    _normalize_moe_output_accumulator,
    _normalize_moe_token_block,
    run_staged_routed_moe_batch,
    run_tiled_staged_routed_moe_batch,
)


class PrefillExecuteError(RuntimeError):
    """Raised when a bounded prefill primitive cannot be executed safely."""


PREFILL_LINEAR_BACKENDS = {"custom-metal", "mpsgraph-f32", "mps-matrix-f32", "auto"}
PREFILL_LINEAR_ACCELERATED_BACKENDS = ("mpsgraph-f32", "mps-matrix-f32")
PREFILL_LINEAR_F32_CONVERSION_BACKENDS = {"mpsgraph-f32", "mps-matrix-f32"}
BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV = (
    "LARGERLM_DISABLE_BATCH_FUSED_ATTN_PROJECTIONS"
)
BATCH_FUSED_ATTN_OUTPUT_DISABLE_ENV = "LARGERLM_DISABLE_BATCH_FUSED_ATTN_OUTPUT"
FUSED_SHARED_EXPERT_BATCH_DISABLE_ENV = "LARGERLM_DISABLE_FUSED_SHARED_EXPERT_BATCH"
FUSED_ROPE_SPLIT_BATCH_DISABLE_ENV = "LARGERLM_DISABLE_FUSED_ROPE_SPLIT_BATCH"
MLA_KEY_CACHE_ENV = "LARGERLM_MLA_KEY_CACHE"
MLA_VALUE_CACHE_DISABLE_ENV = "LARGERLM_MLA_DISABLE_VALUE_CACHE"
MLA_VALUE_CACHE_SINGLETON_ENV = "LARGERLM_MLA_VALUE_CACHE_SINGLETON"
PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD_ENV = (
    "LARGERLM_PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD"
)
PREFILL_LINEAR_CALIBRATION_BACKENDS = (
    "custom-metal",
    "mpsgraph-f32",
    "mps-matrix-f32",
)
PREFILL_LINEAR_MPSGRAPH_DTYPES = {
    "F32",
    "float32",
    "FLOAT32",
    "BF16",
    "bfloat16",
    "BFLOAT16",
    "F16",
    "float16",
    "FLOAT16",
}
PREFILL_ROUTER_GATE_TENSOR_SUFFIX = ".mlp.gate.weight"
CALIBRATION_MATRIX_DTYPE_BYTES = {
    "F32": 4,
    "BF16": 2,
}
PREFILL_CACHE_WRITE_CHUNK_BYTES = 64 * 1024 * 1024
PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV = "LARGERLM_PREFILL_CACHE_WRITE_CHUNK_BYTES"


def is_prompt_prefill_router_gate_tensor(name: object) -> bool:
    return isinstance(name, str) and name.endswith(PREFILL_ROUTER_GATE_TENSOR_SUFFIX)


def is_prompt_prefill_resident_linear_backend_tensor(name: object) -> bool:
    if not isinstance(name, str):
        return False
    if ".layers." not in name:
        return False
    return not is_prompt_prefill_router_gate_tensor(name)


def _batch_fused_attention_projections_enabled() -> bool:
    value = os.environ.get(BATCH_FUSED_ATTN_PROJECTIONS_DISABLE_ENV, "")
    return value.strip().lower() not in {"1", "true", "yes", "on"}


def _batch_fused_attention_output_enabled() -> bool:
    value = os.environ.get(BATCH_FUSED_ATTN_OUTPUT_DISABLE_ENV, "")
    return value.strip().lower() not in {"1", "true", "yes", "on"}


def _fused_shared_expert_batch_enabled() -> bool:
    value = os.environ.get(FUSED_SHARED_EXPERT_BATCH_DISABLE_ENV, "")
    return value.strip().lower() not in {"1", "true", "yes", "on"}


def _fused_rope_split_batch_enabled() -> bool:
    value = os.environ.get(FUSED_ROPE_SPLIT_BATCH_DISABLE_ENV, "")
    return value.strip().lower() not in {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mla_value_cache_disabled_by_env() -> bool:
    return _env_truthy(MLA_VALUE_CACHE_DISABLE_ENV)


def _mla_value_cache_singleton_enabled_by_env() -> bool:
    return _env_truthy(MLA_VALUE_CACHE_SINGLETON_ENV)


AUTO_MPSGRAPH_MIN_BATCH_TOKENS = 2048
AUTO_MPSGRAPH_MIN_DIM = 4096


@dataclass(frozen=True)
class ResidentLinearMatrixScratch:
    matrix_scratch_bytes: int
    matrix_f32_bytes: int
    matrix_raw_conversion_bytes: int


def _resolve_prefill_linear_backend(
    requested: str,
    dtype: str,
    *,
    batch_tokens: int,
    in_dim: int,
    out_dim: int,
    mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
) -> str:
    if requested not in PREFILL_LINEAR_BACKENDS:
        raise PrefillExecuteError(
            "prefill_linear_backend must be custom-metal, mpsgraph-f32, "
            "mps-matrix-f32, or auto"
        )
    if type(mpsgraph_min_batch_tokens) is not int or mpsgraph_min_batch_tokens <= 0:
        raise PrefillExecuteError("prefill_mpsgraph_min_batch_tokens must be positive")
    if type(mpsgraph_min_matrix_dim) is not int or mpsgraph_min_matrix_dim <= 0:
        raise PrefillExecuteError("prefill_mpsgraph_min_matrix_dim must be positive")
    if requested == "auto":
        if (
            dtype in PREFILL_LINEAR_MPSGRAPH_DTYPES
            and batch_tokens >= mpsgraph_min_batch_tokens
            and min(in_dim, out_dim) >= mpsgraph_min_matrix_dim
        ):
            return "mpsgraph-f32"
        return "custom-metal"
    if (
        requested in PREFILL_LINEAR_F32_CONVERSION_BACKENDS
        and dtype not in PREFILL_LINEAR_MPSGRAPH_DTYPES
    ):
        raise PrefillExecuteError(
            f"{requested} backend requires an F32/BF16/F16 resident matrix, got {dtype}"
        )
    return requested


def _positive_integer_mib(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefillExecuteError(f"{name} must be an integer MiB value")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PrefillExecuteError(f"{name} must be finite")
    if parsed <= 0:
        raise PrefillExecuteError(f"{name} must be positive")
    if not parsed.is_integer():
        raise PrefillExecuteError(f"{name} must be an integer MiB value")
    return int(parsed)


def _is_int(value: object) -> bool:
    return type(value) is int


def _require_int(value: object, label: str) -> int:
    if not _is_int(value):
        raise PrefillExecuteError(f"{label} must be an integer")
    return int(value)


def _require_nonnegative_int(value: object, label: str) -> int:
    parsed = _require_int(value, label)
    if parsed < 0:
        raise PrefillExecuteError(f"{label} must be non-negative")
    return parsed


def _require_positive_int(value: object, label: str) -> int:
    parsed = _require_int(value, label)
    if parsed <= 0:
        raise PrefillExecuteError(f"{label} must be positive")
    return parsed


def _optional_positive_int(value: object | None, label: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, label)


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefillExecuteError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PrefillExecuteError(f"{label} must be a finite number")
    return number


def _require_nonnegative_number(value: object, label: str) -> float:
    number = _require_number(value, label)
    if number < 0:
        raise PrefillExecuteError(f"{label} must be non-negative")
    return number


def _require_positive_number(value: object, label: str) -> float:
    number = _require_number(value, label)
    if number <= 0:
        raise PrefillExecuteError(f"{label} must be positive")
    return number


def _optional_positive_number(value: object | None, label: str) -> float | None:
    if value is None:
        return None
    return _require_positive_number(value, label)


def _int_field(tensor: dict[str, Any], field: str, label: str) -> int:
    return _require_int(tensor.get(field), f"{label} {field}")


def _resident_linear_matrix_scratch(
    *,
    matrix_bytes: int,
    dtype: str,
    in_dim: int,
    out_dim: int,
    backend: str,
) -> ResidentLinearMatrixScratch:
    if backend in PREFILL_LINEAR_F32_CONVERSION_BACKENDS:
        matrix_f32_bytes = out_dim * in_dim * 4
        extra_raw = 0 if dtype in {"F32", "float32", "FLOAT32"} else matrix_bytes
        return ResidentLinearMatrixScratch(
            matrix_scratch_bytes=_align_up(matrix_f32_bytes, 2 * 1024 * 1024)
            + extra_raw,
            matrix_f32_bytes=matrix_f32_bytes,
            matrix_raw_conversion_bytes=extra_raw,
        )
    return ResidentLinearMatrixScratch(
        matrix_scratch_bytes=_align_up(matrix_bytes, 2 * 1024 * 1024),
        matrix_f32_bytes=0,
        matrix_raw_conversion_bytes=0,
    )


@dataclass(frozen=True)
class ResidentBatchLinearResult:
    runner_path: Path
    resident_layout_path: Path
    input_path: Path
    output_path: Path
    layer: int
    tensor: str
    tensor_suffix: str
    dtype: str
    backend: str
    batch_tokens: int
    in_dim: int
    out_dim: int
    matrix_bytes: int
    matrix_scratch_bytes: int
    matrix_f32_bytes: int
    matrix_raw_conversion_bytes: int
    input_bytes: int
    output_bytes: int
    estimated_peak_bytes: int
    elapsed_seconds: float
    command: tuple[str, ...]
    runner_backend_elapsed_seconds: float | None = None
    runner_matrix_f32_elapsed_seconds: float | None = None
    runner_accelerator_elapsed_seconds: float | None = None


class ResidentBatchLinearServerSession:
    def __init__(
        self,
        *,
        runner_path: str | Path,
        echo_runner_output: bool = True,
    ) -> None:
        self.runner_path = Path(runner_path)
        if not self.runner_path.exists():
            raise PrefillExecuteError(f"runner not found: {self.runner_path}")
        self.echo_runner_output = echo_runner_output
        self._process: subprocess.Popen[str] | None = None
        self._recent_output: list[str] = []

    def __enter__(self) -> "ResidentBatchLinearServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise PrefillExecuteError("resident batch linear server is already started")
        cmd = [str(self.runner_path), "--run-resident-linear-batch-plan-server-jsonl"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to launch resident batch linear server: {exc}"
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
            success_marker="LargerLM resident batch linear server done",
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

    def submit_batch(
        self,
        *,
        resident_layout_path: str | Path,
        layer: int,
        tensor_suffix: str,
        input_f32_path: str | Path,
        output_f32_path: str | Path,
        batch_tokens: int,
        max_resident_matrix_mib: int = 512,
        max_runner_scratch_mib: int = 4096,
        prefill_linear_backend: str = "custom-metal",
    ) -> str:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise PrefillExecuteError("resident batch linear server stdin is unavailable")
        if process.poll() is not None:
            raise PrefillExecuteError(
                "resident batch linear server is not running"
                + self._recent_output_detail()
            )
        output_path = Path(output_f32_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = {
            "resident_layout": str(Path(resident_layout_path)),
            "layer": int(layer),
            "tensor_suffix": str(tensor_suffix),
            "input_f32": str(Path(input_f32_path)),
            "output_f32": str(output_path),
            "batch_tokens": int(batch_tokens),
            "max_resident_matrix_mib": int(max_resident_matrix_mib),
            "max_runner_scratch_mib": int(max_runner_scratch_mib),
            "prefill_linear_backend": str(prefill_linear_backend),
        }
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to send resident batch linear request to server: {exc}"
            ) from exc
        return self._drain_until(success_marker="  server request:      ok")

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
                    raise PrefillExecuteError(
                        "resident batch linear server stdout closed unexpectedly"
                        + self._recent_output_detail()
                    )
                raise PrefillExecuteError(
                    f"resident batch linear server exited with {status}"
                    + self._recent_output_detail()
                )
            captured.append(line)
            self._record_output(line)
            if self.echo_runner_output:
                print(line, end="")
            if "  server request:      failed" in line:
                raise PrefillExecuteError(
                    "resident batch linear server request failed"
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


class ResidentBatchRMSNormServerSession:
    def __init__(
        self,
        *,
        runner_path: str | Path,
        echo_runner_output: bool = True,
    ) -> None:
        self.runner_path = Path(runner_path)
        if not self.runner_path.exists():
            raise PrefillExecuteError(f"runner not found: {self.runner_path}")
        self.echo_runner_output = echo_runner_output
        self._process: subprocess.Popen[str] | None = None
        self._recent_output: list[str] = []

    def __enter__(self) -> "ResidentBatchRMSNormServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise PrefillExecuteError("resident RMSNorm batch server is already started")
        cmd = [str(self.runner_path), "--run-rmsnorm-batch-server-jsonl"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to launch resident RMSNorm batch server: {exc}"
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
            success_marker="LargerLM resident RMSNorm batch server done",
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

    def submit_batch(
        self,
        *,
        resident_layout_path: str | Path,
        layer: int,
        norm_suffix: str,
        input_f32_path: str | Path,
        output_f32_path: str | Path,
        batch_tokens: int,
        rms_norm_eps: float = 1e-5,
        max_runner_scratch_mib: int = 4096,
    ) -> str:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise PrefillExecuteError("resident RMSNorm batch server stdin is unavailable")
        if process.poll() is not None:
            raise PrefillExecuteError(
                "resident RMSNorm batch server is not running"
                + self._recent_output_detail()
            )
        output_path = Path(output_f32_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request = {
            "resident_layout": str(Path(resident_layout_path)),
            "layer": int(layer),
            "norm_suffix": str(norm_suffix),
            "input_f32": str(Path(input_f32_path)),
            "output_f32": str(output_path),
            "batch_tokens": int(batch_tokens),
            "rms_norm_eps": float(rms_norm_eps),
            "max_runner_scratch_mib": int(max_runner_scratch_mib),
        }
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to send resident RMSNorm batch request to server: {exc}"
            ) from exc
        return self._drain_until(success_marker="  server request:      ok")

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
                    raise PrefillExecuteError(
                        "resident RMSNorm batch server stdout closed unexpectedly"
                        + self._recent_output_detail()
                    )
                raise PrefillExecuteError(
                    f"resident RMSNorm batch server exited with {status}"
                    + self._recent_output_detail()
                )
            captured.append(line)
            self._record_output(line)
            if self.echo_runner_output:
                print(line, end="")
            if "  server request:      failed" in line:
                raise PrefillExecuteError(
                    "resident RMSNorm batch server request failed"
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


class AttentionProjectionsServerSession:
    def __init__(
        self,
        *,
        runner_path: str | Path,
        echo_runner_output: bool = True,
    ) -> None:
        self.runner_path = Path(runner_path)
        if not self.runner_path.exists():
            raise PrefillExecuteError(f"runner not found: {self.runner_path}")
        self.echo_runner_output = echo_runner_output
        self._process: subprocess.Popen[str] | None = None
        self._recent_output: list[str] = []

    def __enter__(self) -> "AttentionProjectionsServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise PrefillExecuteError("attention projections server is already started")
        cmd = [str(self.runner_path), "--run-attn-projections-server-jsonl"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to launch attention projections server: {exc}"
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
            success_marker="LargerLM attention projections server done",
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

    def submit_batch(
        self,
        *,
        resident_layout_path: str | Path,
        layer: int,
        input_f32_path: str | Path,
        output_dir: str | Path,
        batch_tokens: int,
        rms_norm_eps: float = 1e-5,
        max_resident_matrix_mib: int = 512,
        max_runner_scratch_mib: int = 4096,
    ) -> str:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise PrefillExecuteError("attention projections server stdin is unavailable")
        if process.poll() is not None:
            raise PrefillExecuteError(
                "attention projections server is not running"
                + self._recent_output_detail()
            )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        request = {
            "resident_layout": str(Path(resident_layout_path)),
            "layer": int(layer),
            "input_f32": str(Path(input_f32_path)),
            "output_dir": str(output_path),
            "batch_tokens": int(batch_tokens),
            "rms_norm_eps": float(rms_norm_eps),
            "max_resident_matrix_mib": int(max_resident_matrix_mib),
            "max_runner_scratch_mib": int(max_runner_scratch_mib),
        }
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to send attention projections request to server: {exc}"
            ) from exc
        return self._drain_until(success_marker="  server request:      ok")

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
                    raise PrefillExecuteError(
                        "attention projections server stdout closed unexpectedly"
                        + self._recent_output_detail()
                    )
                raise PrefillExecuteError(
                    f"attention projections server exited with {status}"
                    + self._recent_output_detail()
                )
            captured.append(line)
            self._record_output(line)
            if self.echo_runner_output:
                print(line, end="")
            if "  server request:      failed" in line:
                raise PrefillExecuteError(
                    "attention projections server request failed"
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


class AttentionOutputBatchServerSession:
    def __init__(
        self,
        *,
        runner_path: str | Path,
        echo_runner_output: bool = True,
    ) -> None:
        self.runner_path = Path(runner_path)
        if not self.runner_path.exists():
            raise PrefillExecuteError(f"runner not found: {self.runner_path}")
        self.echo_runner_output = echo_runner_output
        self._process: subprocess.Popen[str] | None = None
        self._recent_output: list[str] = []

    def __enter__(self) -> "AttentionOutputBatchServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise PrefillExecuteError(
                "attention output batch server is already started"
            )
        cmd = [str(self.runner_path), "--run-attn-output-batch-server-jsonl"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to launch attention output batch server: {exc}"
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
            success_marker="LargerLM attention output batch server done",
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

    def submit_batch(
        self,
        *,
        resident_layout_path: str | Path,
        layer: int,
        input_f32_path: str | Path,
        residual_f32_path: str | Path,
        output_f32_path: str | Path,
        projection_f32_path: str | Path | None,
        batch_tokens: int,
        max_resident_matrix_mib: int = 512,
        max_runner_scratch_mib: int = 4096,
    ) -> str:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise PrefillExecuteError(
                "attention output batch server stdin is unavailable"
            )
        if process.poll() is not None:
            raise PrefillExecuteError(
                "attention output batch server is not running"
                + self._recent_output_detail()
            )
        output_path = Path(output_f32_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request: dict[str, object] = {
            "resident_layout": str(Path(resident_layout_path)),
            "layer": int(layer),
            "input_f32": str(Path(input_f32_path)),
            "residual_f32": str(Path(residual_f32_path)),
            "output_f32": str(output_path),
            "batch_tokens": int(batch_tokens),
            "max_resident_matrix_mib": int(max_resident_matrix_mib),
            "max_runner_scratch_mib": int(max_runner_scratch_mib),
        }
        if projection_f32_path is not None:
            projection_path = Path(projection_f32_path)
            projection_path.parent.mkdir(parents=True, exist_ok=True)
            request["projection_f32"] = str(projection_path)
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to send attention output batch request to server: {exc}"
            ) from exc
        return self._drain_until(success_marker="  server request:      ok")

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
                    raise PrefillExecuteError(
                        "attention output batch server stdout closed unexpectedly"
                        + self._recent_output_detail()
                    )
                raise PrefillExecuteError(
                    f"attention output batch server exited with {status}"
                    + self._recent_output_detail()
                )
            captured.append(line)
            self._record_output(line)
            if self.echo_runner_output:
                print(line, end="")
            if "  server request:      failed" in line:
                raise PrefillExecuteError(
                    "attention output batch server request failed"
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


class ResidentSharedExpertBatchServerSession:
    def __init__(
        self,
        *,
        runner_path: str | Path,
        echo_runner_output: bool = True,
    ) -> None:
        self.runner_path = Path(runner_path)
        if not self.runner_path.exists():
            raise PrefillExecuteError(f"runner not found: {self.runner_path}")
        self.echo_runner_output = echo_runner_output
        self._process: subprocess.Popen[str] | None = None
        self._recent_output: list[str] = []

    def __enter__(self) -> "ResidentSharedExpertBatchServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise PrefillExecuteError(
                "resident shared expert batch server is already started"
            )
        cmd = [str(self.runner_path), "--run-shared-expert-batch-server-jsonl"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to launch resident shared expert batch server: {exc}"
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
            success_marker="LargerLM resident shared expert batch server done",
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

    def submit_batch(
        self,
        *,
        resident_layout_path: str | Path,
        layer: int,
        input_f32_path: str | Path,
        output_f32_path: str | Path,
        batch_tokens: int,
        max_resident_matrix_mib: int = 512,
        max_runner_scratch_mib: int = 4096,
    ) -> str:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise PrefillExecuteError(
                "resident shared expert batch server stdin is unavailable"
            )
        if process.poll() is not None:
            raise PrefillExecuteError(
                "resident shared expert batch server is not running"
                + self._recent_output_detail()
            )
        output_path = Path(output_f32_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request: dict[str, object] = {
            "resident_layout": str(Path(resident_layout_path)),
            "layer": int(layer),
            "input_f32": str(Path(input_f32_path)),
            "output_f32": str(output_path),
            "batch_tokens": int(batch_tokens),
            "max_resident_matrix_mib": int(max_resident_matrix_mib),
            "max_runner_scratch_mib": int(max_runner_scratch_mib),
        }
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to send shared expert batch request to server: {exc}"
            ) from exc
        return self._drain_until(success_marker="  server request:      ok")

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
                    raise PrefillExecuteError(
                        "resident shared expert batch server stdout closed unexpectedly"
                        + self._recent_output_detail()
                    )
                raise PrefillExecuteError(
                    f"resident shared expert batch server exited with {status}"
                    + self._recent_output_detail()
                )
            captured.append(line)
            self._record_output(line)
            if self.echo_runner_output:
                print(line, end="")
            if "  server request:      failed" in line:
                raise PrefillExecuteError(
                    "resident shared expert batch server request failed"
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


class RopeSplitBatchServerSession:
    def __init__(
        self,
        *,
        runner_path: str | Path,
        echo_runner_output: bool = True,
    ) -> None:
        self.runner_path = Path(runner_path)
        if not self.runner_path.exists():
            raise PrefillExecuteError(f"runner not found: {self.runner_path}")
        self.echo_runner_output = echo_runner_output
        self._process: subprocess.Popen[str] | None = None
        self._recent_output: list[str] = []

    def __enter__(self) -> "RopeSplitBatchServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise PrefillExecuteError("RoPE split batch server is already started")
        cmd = [str(self.runner_path), "--run-rope-split-batch-server-jsonl"]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to launch RoPE split batch server: {exc}"
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
            success_marker="LargerLM RoPE split batch server done",
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

    def submit_batch(
        self,
        *,
        q_b_f32_path: str | Path,
        k_f32_path: str | Path,
        output_q_nope_f32_path: str | Path,
        output_q_rope_f32_path: str | Path,
        output_q_f32_path: str | Path,
        output_k_f32_path: str | Path,
        num_heads: int,
        qk_nope_dim: int,
        rope_dim: int,
        start_position: int,
        batch_tokens: int,
        rope_theta: float = 10000.0,
        rope_interleave: bool = False,
        max_runner_scratch_mib: int = 4096,
    ) -> str:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise PrefillExecuteError("RoPE split batch server stdin is unavailable")
        if process.poll() is not None:
            raise PrefillExecuteError(
                "RoPE split batch server is not running"
                + self._recent_output_detail()
            )
        output_paths = (
            Path(output_q_nope_f32_path),
            Path(output_q_rope_f32_path),
            Path(output_q_f32_path),
            Path(output_k_f32_path),
        )
        for output_path in output_paths:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        request = {
            "q_b_f32": str(Path(q_b_f32_path)),
            "k_f32": str(Path(k_f32_path)),
            "output_q_nope_f32": str(output_paths[0]),
            "output_q_rope_f32": str(output_paths[1]),
            "output_q_f32": str(output_paths[2]),
            "output_k_f32": str(output_paths[3]),
            "num_heads": int(num_heads),
            "qk_nope_dim": int(qk_nope_dim),
            "rope_dim": int(rope_dim),
            "start_position": int(start_position),
            "batch_tokens": int(batch_tokens),
            "rope_theta": float(rope_theta),
            "rope_interleave": bool(rope_interleave),
            "max_runner_scratch_mib": int(max_runner_scratch_mib),
        }
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to send RoPE split batch request to server: {exc}"
            ) from exc
        return self._drain_until(success_marker="  server request:      ok")

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
                    raise PrefillExecuteError(
                        "RoPE split batch server stdout closed unexpectedly"
                        + self._recent_output_detail()
                    )
                raise PrefillExecuteError(
                    f"RoPE split batch server exited with {status}"
                    + self._recent_output_detail()
                )
            captured.append(line)
            self._record_output(line)
            if self.echo_runner_output:
                print(line, end="")
            if "  server request:      failed" in line:
                raise PrefillExecuteError(
                    "RoPE split batch server request failed"
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


class MLAAttentionBatchServerSession:
    def __init__(
        self,
        *,
        runner_path: str | Path,
        echo_runner_output: bool = True,
        mla_key_cache: bool | None = None,
        disable_value_cache: bool | None = None,
    ) -> None:
        self.runner_path = Path(runner_path)
        if not self.runner_path.exists():
            raise PrefillExecuteError(f"runner not found: {self.runner_path}")
        if mla_key_cache is not None and type(mla_key_cache) is not bool:
            raise PrefillExecuteError("mla_key_cache must be a boolean")
        if disable_value_cache is not None and type(disable_value_cache) is not bool:
            raise PrefillExecuteError("disable_value_cache must be a boolean")
        self.echo_runner_output = echo_runner_output
        self.mla_key_cache = (
            _env_truthy(MLA_KEY_CACHE_ENV) if mla_key_cache is None else mla_key_cache
        )
        self.disable_value_cache = (
            _mla_value_cache_disabled_by_env()
            if disable_value_cache is None
            else disable_value_cache
        )
        self._process: subprocess.Popen[str] | None = None
        self._recent_output: list[str] = []

    def __enter__(self) -> "MLAAttentionBatchServerSession":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def is_compatible(self, *, mla_key_cache: bool, disable_value_cache: bool) -> bool:
        return (
            self.mla_key_cache is bool(mla_key_cache)
            and self.disable_value_cache is bool(disable_value_cache)
        )

    def start(self) -> None:
        if self._process is not None:
            raise PrefillExecuteError("MLA attention batch server is already started")
        cmd = [str(self.runner_path), "--run-mla-attention-batch-server-jsonl"]
        env = os.environ.copy()
        if self.mla_key_cache:
            env[MLA_KEY_CACHE_ENV] = "1"
        else:
            env.pop(MLA_KEY_CACHE_ENV, None)
        if self.disable_value_cache:
            env[MLA_VALUE_CACHE_DISABLE_ENV] = "1"
        else:
            env.pop(MLA_VALUE_CACHE_DISABLE_ENV, None)
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to launch MLA attention batch server: {exc}"
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
            success_marker="LargerLM MLA attention batch server done",
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

    def submit_batch(
        self,
        *,
        resident_layout_path: str | Path,
        cache_layout_path: str | Path,
        cache_file_path: str | Path,
        layer: int,
        q_nope_f32_path: str | Path,
        q_rope_f32_path: str | Path,
        output_f32_path: str | Path,
        context_length: int,
        start_position: int,
        batch_tokens: int,
        num_heads: int,
        qk_nope_dim: int,
        rope_dim: int,
        v_head_dim: int,
        mla_kv_b_cache_dir: str | Path | None = None,
        indices_u32_path: str | Path | None = None,
        index_topk: int | None = None,
        kv_lora_dim: int | None = None,
        cache_position_offset: int = 0,
        attention_scale: float | None = None,
        rope_theta: float = 10000.0,
        rope_interleave: bool = False,
        max_cache_file_mib: int = 32768,
        max_cache_read_mib: int = 256,
        max_resident_matrix_mib: int = 512,
        max_runner_scratch_mib: int = 4096,
    ) -> str:
        if self._process is None:
            self.start()
        process = self._process
        if process is None or process.stdin is None:
            raise PrefillExecuteError("MLA attention batch server stdin is unavailable")
        if process.poll() is not None:
            raise PrefillExecuteError(
                "MLA attention batch server is not running"
                + self._recent_output_detail()
            )
        output_path = Path(output_f32_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        request: dict[str, object] = {
            "resident_layout": str(Path(resident_layout_path)),
            "cache_layout": str(Path(cache_layout_path)),
            "cache_file": str(Path(cache_file_path)),
            "layer": int(layer),
            "q_nope_f32": str(Path(q_nope_f32_path)),
            "q_rope_f32": str(Path(q_rope_f32_path)),
            "output_f32": str(output_path),
            "context_length": int(context_length),
            "batch_tokens": int(batch_tokens),
            "num_heads": int(num_heads),
            "qk_nope_dim": int(qk_nope_dim),
            "rope_dim": int(rope_dim),
            "v_head_dim": int(v_head_dim),
            "cache_position_offset": int(cache_position_offset),
            "rope_theta": float(rope_theta),
            "rope_interleave": bool(rope_interleave),
            "max_cache_file_mib": int(max_cache_file_mib),
            "max_cache_read_mib": int(max_cache_read_mib),
            "max_resident_matrix_mib": int(max_resident_matrix_mib),
            "max_runner_scratch_mib": int(max_runner_scratch_mib),
        }
        if kv_lora_dim is not None:
            request["kv_lora_dim"] = int(kv_lora_dim)
        if attention_scale is not None:
            request["attention_scale"] = float(attention_scale)
        if mla_kv_b_cache_dir is not None:
            request["mla_kv_b_cache_dir"] = str(Path(mla_kv_b_cache_dir))
        if indices_u32_path is not None:
            if index_topk is None:
                raise PrefillExecuteError(
                    "index_topk must be provided with indices_u32_path"
                )
            request["indices_u32"] = str(Path(indices_u32_path))
            request["index_topk"] = int(index_topk)
        else:
            request["start_position"] = int(start_position)
        try:
            process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to send MLA attention batch request to server: {exc}"
            ) from exc
        return self._drain_until(success_marker="  server request:      ok")

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
                    raise PrefillExecuteError(
                        "MLA attention batch server stdout closed unexpectedly"
                        + self._recent_output_detail()
                    )
                raise PrefillExecuteError(
                    f"MLA attention batch server exited with {status}"
                    + self._recent_output_detail()
                )
            captured.append(line)
            self._record_output(line)
            if self.echo_runner_output:
                print(line, end="")
            if "  server request:      failed" in line:
                raise PrefillExecuteError(
                    "MLA attention batch server request failed"
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


@dataclass(frozen=True)
class ResidentSharedExpertBatchResult:
    runner_path: Path
    resident_layout_path: Path
    input_path: Path
    output_path: Path
    layer: int
    name: str
    tensor: str
    dtype: str
    backend: str
    batch_tokens: int
    hidden_dim: int
    intermediate_dim: int
    gate_tensor: str
    up_tensor: str
    down_tensor: str
    gate_matrix_bytes: int
    up_matrix_bytes: int
    down_matrix_bytes: int
    matrix_bytes: int
    matrix_scratch_bytes: int
    input_bytes: int
    activation_bytes: int
    output_bytes: int
    estimated_peak_bytes: int
    elapsed_seconds: float
    command: tuple[str, ...]
    runner_backend_elapsed_seconds: float | None = None


@dataclass(frozen=True)
class ResidentLinearCalibrationCase:
    batch_tokens: int
    matrix_dim: int
    matrix_bytes: int
    input_bytes: int
    output_bytes: int
    estimated_flops: int
    custom_elapsed_seconds: float
    mpsgraph_elapsed_seconds: float
    mpsgraph_speedup: float
    mpsgraph_meets_threshold: bool
    winner: str
    custom_estimated_peak_bytes: int
    mpsgraph_estimated_peak_bytes: int
    in_dim: int = 0
    out_dim: int = 0
    min_matrix_dim: int = 0
    mps_matrix_elapsed_seconds: float = 0.0
    mps_matrix_speedup: float = 0.0
    mps_matrix_estimated_peak_bytes: int = 0
    backend_elapsed_seconds: dict[str, float] | None = None
    backend_estimated_peak_bytes: dict[str, int] | None = None
    matrix_dtype: str = "F32"


@dataclass(frozen=True)
class ResidentLinearCalibrationWorkDirBudget:
    source: str
    case_count: int
    batch_token_count: int
    shape_count: int
    repeats: int
    backend_count: int
    calibrated_backends: tuple[str, ...]
    backend_output_file_count: int
    matrix_bytes: int
    input_bytes: int
    single_backend_output_bytes: int
    total_backend_output_bytes: int
    single_case_bytes: int
    estimated_work_dir_bytes: int
    max_calibration_work_dir_bytes: int
    max_calibration_work_dir_mib: int
    disk_usage_path: Path | None = None
    disk_available_bytes: int | None = None
    disk_required_bytes: int | None = None
    disk_safety_margin_bytes: int = 0


@dataclass(frozen=True)
class ResidentLinearCalibrationResult:
    runner_path: Path
    work_dir: Path
    kept_work_dir: bool
    batch_token_values: tuple[int, ...]
    matrix_dim_values: tuple[int, ...]
    repeats: int
    min_mpsgraph_speedup: float
    max_calibration_case_bytes: int
    max_resident_matrix_mib: int
    max_runner_scratch_mib: int
    recommended_prefill_mpsgraph_min_batch_tokens: int | None
    recommended_prefill_mpsgraph_min_matrix_dim: int | None
    suggested_prefill_runtime_policy_flags: dict[str, object] | None
    suggested_launch_profile: dict[str, object] | None
    cases: tuple[ResidentLinearCalibrationCase, ...]
    matrix_shapes: tuple[tuple[int, int], ...] = ()
    work_dir_budget: ResidentLinearCalibrationWorkDirBudget | None = None
    backend_comparison: dict[str, object] | None = None
    matrix_dtype: str = "F32"


@dataclass(frozen=True)
class ResidentBatchRMSNormResult:
    runner_path: Path
    resident_layout_path: Path
    input_path: Path
    output_path: Path
    layer: int
    tensor: str
    norm_suffix: str
    dtype: str
    batch_tokens: int
    hidden_dim: int
    vector_bytes: int
    input_bytes: int
    output_bytes: int
    estimated_peak_bytes: int
    rms_norm_eps: float
    command: tuple[str, ...]


@dataclass(frozen=True)
class PrefillAttentionPrefixResult:
    runner_path: Path
    resident_layout_path: Path
    input_path: Path
    output_dir: Path
    layer: int
    batch_tokens: int
    hidden_dim: int
    q_a_dim: int
    kv_a_dim: int
    input_bytes: int
    norm_output_bytes: int
    q_a_output_bytes: int
    kv_a_output_bytes: int
    estimated_peak_bytes: int
    input_layernorm: ResidentBatchRMSNormResult
    q_a_proj: ResidentBatchLinearResult
    kv_a_proj_with_mqa: ResidentBatchLinearResult


@dataclass(frozen=True)
class PrefillAttentionProjectionBatchResult:
    runner_path: Path
    resident_layout_path: Path
    input_path: Path
    output_dir: Path
    layer: int
    batch_tokens: int
    hidden_dim: int
    q_a_dim: int
    q_b_dim: int
    kv_a_dim: int
    kv_lora_dim: int
    kv_rope_dim: int
    kv_b_dim: int
    attention_value_source: str
    input_bytes: int
    q_b_output_bytes: int
    kv_a_lora_bytes: int
    kv_a_rope_bytes: int
    kv_b_output_bytes: int
    split_peak_bytes: int
    estimated_peak_bytes: int
    prefix: PrefillAttentionPrefixResult
    q_a_layernorm: ResidentBatchRMSNormResult
    q_b_proj: ResidentBatchLinearResult
    kv_a_lora_path: Path
    kv_a_rope_path: Path
    kv_a_layernorm: ResidentBatchRMSNormResult
    kv_b_proj: ResidentBatchLinearResult | None


@dataclass(frozen=True)
class PrefillCacheWriteResult:
    cache_layout_path: Path
    cache_file_path: Path
    input_path: Path
    layer: int
    start_position: int
    batch_tokens: int
    width: int
    dtype: str
    dtype_bytes: int
    token_stride_bytes: int
    segment_offset: int
    first_write_offset: int
    input_bytes: int
    encoded_bytes: int
    max_cache_file_bytes: int
    max_cache_write_bytes: int
    write_chunk_tokens: int
    write_chunk_bytes: int
    write_chunks: int
    estimated_peak_bytes: int
    encoder: str


@dataclass(frozen=True)
class PrefillRopeBatchResult:
    runner_path: Path
    q_b_input_path: Path
    k_rope_input_path: Path
    output_dir: Path
    q_nope_path: Path
    q_rope_path: Path
    q_rope_rotated_path: Path
    k_rope_rotated_path: Path
    batch_tokens: int
    num_heads: int
    qk_nope_dim: int
    rope_dim: int
    start_position: int
    rope_theta: float
    rope_interleave: bool
    q_b_input_bytes: int
    k_rope_input_bytes: int
    q_nope_bytes: int
    q_rope_bytes: int
    k_rope_bytes: int
    split_peak_bytes: int
    estimated_peak_bytes: int
    command: tuple[str, ...]


@dataclass(frozen=True)
class PrefillMLAAttentionBatchResult:
    runner_path: Path
    resident_layout_path: Path
    cache_layout_path: Path
    cache_file_path: Path
    q_nope_path: Path
    q_rope_path: Path
    indices_u32_path: Path | None
    output_path: Path
    mla_kv_b_cache_dir: Path | None
    layer: int
    context_length: int
    start_position: int
    batch_tokens: int
    num_heads: int
    kv_lora_dim: int
    qk_nope_dim: int
    rope_dim: int
    v_head_dim: int
    cache_position_offset: int
    indexed: bool
    index_topk: int | None
    attention_scale: float
    rope_theta: float
    rope_interleave: bool
    attention_value_source: str
    q_nope_bytes: int
    q_rope_bytes: int
    indices_u32_bytes: int
    output_bytes: int
    cache_read_bytes: int
    cache_f32_bytes: int
    kv_b_matrix_bytes: int
    kv_b_f32_bytes: int
    mla_key_cache: bool
    mla_key_cache_bytes: int
    mla_value_cache: bool
    mla_value_cache_bytes: int
    estimated_peak_bytes: int
    command: tuple[str, ...]
    mla_timing_input_elapsed_seconds: float | None = None
    mla_timing_cache_read_elapsed_seconds: float | None = None
    mla_timing_value_read_elapsed_seconds: float | None = None
    mla_timing_metal_setup_elapsed_seconds: float | None = None
    mla_timing_kernel_elapsed_seconds: float | None = None
    mla_timing_write_elapsed_seconds: float | None = None
    mla_timing_total_elapsed_seconds: float | None = None


@dataclass(frozen=True)
class PrefillAttentionOutputBatchResult:
    runner_path: Path
    resident_layout_path: Path
    attn_value_path: Path
    residual_path: Path
    projection_path: Path
    output_path: Path
    layer: int
    batch_tokens: int
    attn_value_dim: int
    hidden_dim: int
    attn_value_bytes: int
    residual_bytes: int
    projection_bytes: int
    output_bytes: int
    residual_add_peak_bytes: int
    estimated_peak_bytes: int
    o_proj: ResidentBatchLinearResult


@dataclass(frozen=True)
class PrefillAttentionBlockBatchResult:
    runner_path: Path
    resident_layout_path: Path
    cache_layout_path: Path
    cache_file_path: Path
    input_path: Path
    output_dir: Path
    output_path: Path
    layer: int
    context_length: int
    start_position: int
    batch_tokens: int
    num_heads: int
    kv_lora_dim: int
    qk_nope_dim: int
    rope_dim: int
    v_head_dim: int
    hidden_dim: int
    mla_key_cache: bool
    mla_key_cache_bytes: int
    mla_value_cache: bool
    mla_value_cache_bytes: int
    input_bytes: int
    output_bytes: int
    cache_write_peak_bytes: int
    estimated_peak_bytes: int
    projections_elapsed_seconds: float
    cache_write_elapsed_seconds: float
    rope_elapsed_seconds: float
    dsa_indexer_elapsed_seconds: float
    mla_attention_elapsed_seconds: float
    attention_output_elapsed_seconds: float
    projections: PrefillAttentionProjectionBatchResult
    cache_write: PrefillCacheWriteResult
    dsa_indexer: DSAIndexerBatchResult | None
    dsa_rope_interleave: bool
    dsa_indices_u32_path: Path | None
    rope: PrefillRopeBatchResult
    mla_attention: PrefillMLAAttentionBatchResult
    attention_output: PrefillAttentionOutputBatchResult


@dataclass(frozen=True)
class PrefillDenseMLPBlockBatchResult:
    runner_path: Path
    resident_layout_path: Path
    input_path: Path
    output_dir: Path
    output_path: Path
    layer: int
    batch_tokens: int
    hidden_dim: int
    intermediate_dim: int
    input_bytes: int
    norm_output_bytes: int
    gate_output_bytes: int
    up_output_bytes: int
    swiglu_output_bytes: int
    down_output_bytes: int
    output_bytes: int
    swiglu_peak_bytes: int
    residual_add_peak_bytes: int
    estimated_peak_bytes: int
    post_attention_layernorm: ResidentBatchRMSNormResult
    gate_proj: ResidentBatchLinearResult
    up_proj: ResidentBatchLinearResult
    down_proj: ResidentBatchLinearResult


@dataclass(frozen=True)
class PrefillRoutedMLPBlockBatchResult:
    runner_path: Path
    expert_layout_path: Path
    resident_layout_path: Path
    input_path: Path
    output_dir: Path
    output_path: Path
    router_json_dir: Path | None
    layer: int
    batch_tokens: int
    hidden_dim: int
    input_bytes: int
    output_bytes: int
    token_input_bytes: int
    token_output_bytes: int
    top_k: int
    max_k: int
    router_score: str
    include_shared_expert: bool
    rms_norm_eps: float
    estimated_peak_bytes: int
    read_bytes: int
    command_count: int
    first_command: tuple[str, ...]


@dataclass(frozen=True)
class PrefillStagedRoutedMLPBlockBatchResult:
    runner_path: Path
    expert_layout_path: Path
    resident_layout_path: Path
    input_path: Path
    output_dir: Path
    output_path: Path
    router_json_dir: Path
    stage_file_path: Path
    stage_manifest_path: Path
    routed_output_path: Path
    layer: int
    batch_tokens: int
    hidden_dim: int
    input_bytes: int
    norm_output_bytes: int
    routed_output_bytes: int
    output_bytes: int
    top_k: int
    max_k: int
    router_score: str
    include_shared_expert: bool
    rms_norm_eps: float
    staged_bytes: int
    compact_stage_bytes: int
    compact_stage_materialized_bytes: int
    compact_stage_storage: str
    stage_plus_compact_bytes: int
    stage_plus_compact_materialized_bytes: int
    static_capacity_path: Path | None
    static_capacity_binary_path: Path | None
    static_capacity_per_expert: int | None
    static_capacity_used_slots: int
    static_capacity_total_slots: int
    static_capacity_overflow_assignments: int
    static_capacity_binary_bytes: int
    shared_output_path: Path | None
    shared_output_bytes: int
    shared_swiglu_peak_bytes: int
    shared_add_peak_bytes: int
    residual_add_peak_bytes: int
    estimated_peak_bytes: int
    router_command_count: int
    routed_command_count: int
    first_router_command: tuple[str, ...]
    router_margin_summary: dict[str, Any] | None
    post_attention_layernorm: ResidentBatchRMSNormResult
    router_gate_proj: ResidentBatchLinearResult | None
    stage_result: BatchExpertStageResult
    staged_moe: StagedRoutedMoEBatchResult
    tiled_staged_moe: TiledStagedRoutedMoEBatchResult | None
    shared_expert_batch: ResidentSharedExpertBatchResult | None
    shared_gate_proj: ResidentBatchLinearResult | None
    shared_up_proj: ResidentBatchLinearResult | None
    shared_down_proj: ResidentBatchLinearResult | None
    stage_copy_elapsed_seconds: float | None = None
    stage_copy_throughput_gib_per_second: float | None = None
    stage_copy_seconds_ok: bool | None = None
    routed_moe_elapsed_seconds: float | None = None


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PrefillExecuteError(f"failed to read resident layout {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PrefillExecuteError(f"failed to parse resident layout {p}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrefillExecuteError(f"resident layout {p} must be a JSON object")
    return payload


def _dtype_bytes(dtype: str) -> int:
    if dtype in {"F32", "float32"}:
        return 4
    if dtype in {"BF16", "bfloat16", "F16", "float16"}:
        return 2
    return 0


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _tensor_shape2(tensor: dict[str, Any], label: str) -> tuple[int, int]:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or not _is_int(shape[0])
        or not _is_int(shape[1])
    ):
        raise PrefillExecuteError(f"{label} must have shape [rows, cols]")
    rows = int(shape[0])
    cols = int(shape[1])
    if rows <= 0 or cols <= 0:
        raise PrefillExecuteError(f"{label} shape must be positive")
    dtype = str(tensor.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise PrefillExecuteError(f"unsupported dtype {dtype} for {label}")
    size = _int_field(tensor, "size", label)
    expected = rows * cols * dtype_nbytes
    if size != expected:
        raise PrefillExecuteError(f"{label} size {size} does not match expected {expected}")
    return rows, cols


def _tensor_shape1(tensor: dict[str, Any], label: str) -> int:
    shape = tensor.get("shape")
    if not isinstance(shape, list) or len(shape) < 1 or not _is_int(shape[0]):
        raise PrefillExecuteError(f"{label} must have shape [dim]")
    dim = int(shape[0])
    if dim <= 0:
        raise PrefillExecuteError(f"{label} dim must be positive")
    dtype = str(tensor.get("dtype") or "")
    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise PrefillExecuteError(f"unsupported dtype {dtype} for {label}")
    size = _int_field(tensor, "size", label)
    expected = dim * dtype_nbytes
    if size != expected:
        raise PrefillExecuteError(f"{label} size {size} does not match expected {expected}")
    return dim


def _tensors(layout: dict[str, Any]) -> list[dict[str, Any]]:
    tensors = layout.get("tensors")
    if not isinstance(tensors, list):
        raise PrefillExecuteError("resident layout missing tensors array")
    return [tensor for tensor in tensors if isinstance(tensor, dict)]


def _check_tensor_backing_span(
    weight_path: Path,
    tensor: dict[str, Any],
    label: str,
) -> None:
    offset = tensor.get("offset")
    size = tensor.get("size")
    if type(offset) is not int or offset < 0 or type(size) is not int or size < 0:
        raise PrefillExecuteError(f"{label} must have non-negative offset and size")
    try:
        file_bytes = weight_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(
            f"failed to stat resident weight file {weight_path}: {exc}"
        ) from exc
    end = int(offset) + int(size)
    if end > file_bytes:
        raise PrefillExecuteError(
            f"{label} extends beyond resident weight file: {end} > {file_bytes}"
        )


def _find_layer_matrix(
    layout: dict[str, Any],
    *,
    layer: int,
    suffix: str,
) -> dict[str, Any]:
    layer = _require_int(layer, "layer")
    if layer < 0:
        raise PrefillExecuteError("layer must be non-negative")
    if not suffix:
        raise PrefillExecuteError("tensor suffix must be non-empty")
    needle = f".layers.{layer}."
    matches: list[dict[str, Any]] = []
    for tensor in _tensors(layout):
        name = tensor.get("name")
        if isinstance(name, str) and needle in name and name.endswith(suffix):
            matches.append(tensor)
    if not matches:
        raise PrefillExecuteError(
            f"resident matrix with suffix {suffix} for layer {layer} not found"
        )
    matches.sort(key=lambda item: len(str(item.get("name") or "")))
    return matches[0]


def _find_layer_tensor_optional(
    layout: dict[str, Any],
    *,
    layer: int,
    suffix: str,
) -> dict[str, Any] | None:
    try:
        return _find_layer_matrix(layout, layer=layer, suffix=suffix)
    except PrefillExecuteError:
        return None


def _find_tensor_by_name(
    layout: dict[str, Any],
    name: str,
) -> dict[str, Any] | None:
    for tensor in _tensors(layout):
        if tensor.get("name") == name:
            return tensor
    return None


def _find_layer_vector(
    layout: dict[str, Any],
    *,
    layer: int,
    suffix: str,
) -> dict[str, Any]:
    layer = _require_int(layer, "layer")
    if layer < 0:
        raise PrefillExecuteError("layer must be non-negative")
    if not suffix:
        raise PrefillExecuteError("norm suffix must be non-empty")
    needle = f".layers.{layer}."
    matches: list[dict[str, Any]] = []
    for tensor in _tensors(layout):
        name = tensor.get("name")
        if isinstance(name, str) and needle in name and name.endswith(suffix):
            matches.append(tensor)
    if not matches:
        raise PrefillExecuteError(
            f"resident vector with suffix {suffix} for layer {layer} not found"
        )
    matches.sort(key=lambda item: len(str(item.get("name") or "")))
    return matches[0]


def _tensor_shape3_value_source(
    layout: dict[str, Any],
    tensor: dict[str, Any],
    label: str,
    weight_path: Path,
) -> tuple[int, int, int, int, int]:
    shape = tensor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or not all(_is_int(dim) and int(dim) > 0 for dim in shape)
    ):
        raise PrefillExecuteError(f"{label} must have a 3-D positive integer shape")
    d0, d1, d2 = int(shape[0]), int(shape[1]), int(shape[2])
    dtype = str(tensor.get("dtype") or "")
    size = _int_field(tensor, "size", label)
    if dtype in {"U32", "uint32", "UINT32"}:
        name = tensor.get("name")
        if not isinstance(name, str) or not name.endswith(".weight"):
            raise PrefillExecuteError(f"{label} MXFP4 tensor must end with .weight")
        scale_name = name[:-7] + ".scales"
        scales = _find_tensor_by_name(layout, scale_name)
        if scales is None:
            raise PrefillExecuteError(f"{label} MXFP4 tensor missing {scale_name}")
        scale_dtype = str(scales.get("dtype") or "")
        if scale_dtype not in {"U8", "uint8", "UINT8"}:
            raise PrefillExecuteError(f"{label} MXFP4 scales must have U8 dtype")
        scale_shape = scales.get("shape")
        if (
            not isinstance(scale_shape, list)
            or len(scale_shape) != 3
            or not all(_is_int(dim) and int(dim) > 0 for dim in scale_shape)
        ):
            raise PrefillExecuteError(f"{label}.scales must have a 3-D shape")
        s0, s1, groups = int(scale_shape[0]), int(scale_shape[1]), int(scale_shape[2])
        if s0 != d0 or s1 != d1:
            raise PrefillExecuteError(
                f"{label} MXFP4 scales must match first two weight dims"
            )
        logical_d2 = d2 * 8
        if logical_d2 % groups != 0:
            raise PrefillExecuteError(
                f"{label} MXFP4 groups do not divide logical dim {logical_d2}"
            )
        group_size = logical_d2 // groups
        if group_size <= 0 or group_size % 8 != 0:
            raise PrefillExecuteError(
                f"{label} MXFP4 group size must be a positive multiple of 8"
            )
        expected_weight = d0 * d1 * d2 * 4
        scale_size = _int_field(scales, "size", f"{label}.scales")
        expected_scales = d0 * d1 * groups
        if size != expected_weight or scale_size != expected_scales:
            raise PrefillExecuteError(
                f"{label} MXFP4 size mismatch: weight {size}/{expected_weight}, "
                f"scales {scale_size}/{expected_scales}"
            )
        _check_tensor_backing_span(weight_path, tensor, label)
        _check_tensor_backing_span(weight_path, scales, f"{label}.scales")
        return d0, d1, logical_d2, size + scale_size, d0 * d1 * logical_d2 * 4

    dtype_nbytes = _dtype_bytes(dtype)
    if dtype_nbytes == 0:
        raise PrefillExecuteError(f"unsupported dtype {dtype} for {label}")
    expected = d0 * d1 * d2 * dtype_nbytes
    if size != expected:
        raise PrefillExecuteError(f"{label} size {size} does not match expected {expected}")
    _check_tensor_backing_span(weight_path, tensor, label)
    return d0, d1, d2, size, d0 * d1 * d2 * 4


def _resolve_layer_matrix_suffix(
    layout_path: str | Path,
    *,
    layer: int,
    suffixes: tuple[str, ...],
) -> str:
    layout = _load_json(layout_path)
    errors: list[str] = []
    for suffix in suffixes:
        try:
            _find_layer_matrix(layout, layer=layer, suffix=suffix)
            return suffix
        except PrefillExecuteError as exc:
            errors.append(str(exc))
    joined = "; ".join(errors)
    raise PrefillExecuteError(f"none of the resident matrix suffixes matched: {joined}")


def _run_command(
    cmd: list[str],
    *,
    echo_output: bool,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, text=True, capture_output=True, env=env)
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
        raise PrefillExecuteError(
            f"prefill runner command failed with exit {completed.returncode}: "
            f"{' '.join(cmd)}{detail}"
        )
    return completed


def _parse_runner_timing_seconds(stdout: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("timing ") or ":" not in stripped:
            continue
        label, raw_value = stripped.split(":", 1)
        key = label[len("timing ") :].strip().replace(" ", "_")
        try:
            value = float(raw_value.strip())
        except ValueError:
            continue
        if math.isfinite(value) and value >= 0.0:
            timings[key] = value
    return timings


def _split_f32_row_prefix(
    *,
    input_path: Path,
    prefix_output_path: Path,
    suffix_output_path: Path,
    batch_tokens: int,
    row_dim: int,
    prefix_dim: int,
) -> tuple[int, int, int]:
    if batch_tokens <= 0 or row_dim <= 0 or prefix_dim <= 0:
        raise PrefillExecuteError("split dimensions must be positive")
    if prefix_dim > row_dim:
        raise PrefillExecuteError(
            f"prefix dim {prefix_dim} cannot exceed row dim {row_dim}"
        )
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat split input {input_path}: {exc}") from exc
    row_bytes = row_dim * 4
    prefix_row_bytes = prefix_dim * 4
    suffix_row_bytes = row_bytes - prefix_row_bytes
    expected_input_bytes = batch_tokens * row_bytes
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"split input bytes {input_bytes} do not match expected "
            f"{expected_input_bytes}"
        )
    prefix_output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open("rb") as source, prefix_output_path.open(
            "wb"
        ) as prefix_out, suffix_output_path.open("wb") as suffix_out:
            for _ in range(batch_tokens):
                row = source.read(row_bytes)
                if len(row) != row_bytes:
                    raise PrefillExecuteError(
                        f"failed to read full f32 split row from {input_path}"
                    )
                prefix_out.write(row[:prefix_row_bytes])
                suffix_out.write(row[prefix_row_bytes:])
    except OSError as exc:
        raise PrefillExecuteError(f"failed to split f32 rows from {input_path}: {exc}") from exc
    return (
        batch_tokens * prefix_row_bytes,
        batch_tokens * suffix_row_bytes,
        row_bytes,
    )


def _f32_to_bf16_bits(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _encode_f32_payload_as_bf16(payload_f32: bytes) -> bytes:
    values = array("I")
    if values.itemsize != 4:
        count = len(payload_f32) // 4
        unpacked = struct.unpack(f"<{count}I", payload_f32)
        encoded = array(
            "H",
            (
                ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
                for bits in unpacked
            ),
        )
    else:
        values.frombytes(payload_f32)
        if sys.byteorder != "little":
            values.byteswap()
        encoded = array(
            "H",
            (
                ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
                for bits in values
            ),
        )
    if sys.byteorder != "little":
        encoded.byteswap()
    return encoded.tobytes()


def _encode_cache_rows(
    payload_f32: bytes,
    *,
    width: int,
    batch_tokens: int,
    dtype: str,
    dtype_bytes: int,
) -> bytes:
    expected_f32_bytes = batch_tokens * width * 4
    if len(payload_f32) != expected_f32_bytes:
        raise PrefillExecuteError(
            f"cache input chunk has {len(payload_f32)} bytes, expected "
            f"{expected_f32_bytes}"
        )
    if dtype in {"F32", "float32"}:
        if dtype_bytes != 4:
            raise PrefillExecuteError("cache dtype_bytes does not match F32")
        return payload_f32
    if dtype in {"BF16", "bfloat16"}:
        if dtype_bytes != 2:
            raise PrefillExecuteError("cache dtype_bytes does not match BF16")
        return _encode_f32_payload_as_bf16(payload_f32)
    raise PrefillExecuteError(f"prefill cache write supports BF16 or F32, got {dtype}")


def _encode_cache_row(row_f32: bytes, *, width: int, dtype: str, dtype_bytes: int) -> bytes:
    return _encode_cache_rows(
        row_f32,
        width=width,
        batch_tokens=1,
        dtype=dtype,
        dtype_bytes=dtype_bytes,
    )


def _prefill_cache_write_chunk_bytes() -> int:
    raw = os.environ.get(PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV)
    if raw is None or raw == "":
        return PREFILL_CACHE_WRITE_CHUNK_BYTES
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise PrefillExecuteError(
            f"{PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV} must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise PrefillExecuteError(
            f"{PREFILL_CACHE_WRITE_CHUNK_BYTES_ENV} must be a positive integer"
        )
    return parsed


def _split_q_b_batch(
    *,
    input_path: Path,
    q_nope_path: Path,
    q_rope_path: Path,
    batch_tokens: int,
    num_heads: int,
    qk_nope_dim: int,
    rope_dim: int,
) -> tuple[int, int, int]:
    if batch_tokens <= 0 or num_heads <= 0 or qk_nope_dim <= 0 or rope_dim <= 0:
        raise PrefillExecuteError("q_b split dimensions must be positive")
    head_dim = qk_nope_dim + rope_dim
    row_bytes = num_heads * head_dim * 4
    nope_head_bytes = qk_nope_dim * 4
    rope_head_bytes = rope_dim * 4
    expected_input_bytes = batch_tokens * row_bytes
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat q_b input {input_path}: {exc}") from exc
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"q_b input bytes {input_bytes} do not match expected {expected_input_bytes}"
        )
    q_nope_path.parent.mkdir(parents=True, exist_ok=True)
    q_rope_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open("rb") as source, q_nope_path.open(
            "wb"
        ) as nope_out, q_rope_path.open("wb") as rope_out:
            for _ in range(batch_tokens):
                row = source.read(row_bytes)
                if len(row) != row_bytes:
                    raise PrefillExecuteError(
                        f"failed to read full q_b row from {input_path}"
                    )
                for head in range(num_heads):
                    base = head * head_dim * 4
                    nope_out.write(row[base : base + nope_head_bytes])
                    rope_base = base + nope_head_bytes
                    rope_out.write(row[rope_base : rope_base + rope_head_bytes])
    except OSError as exc:
        raise PrefillExecuteError(f"failed to split q_b batch rows: {exc}") from exc
    return (
        batch_tokens * num_heads * nope_head_bytes,
        batch_tokens * num_heads * rope_head_bytes,
        row_bytes,
    )


def _apply_rope_to_f32_array(
    values: array,
    *,
    vector_count: int,
    rope_dim: int,
    position: int,
    theta: float,
    interleave: bool,
) -> None:
    half_dim = rope_dim // 2
    for vector in range(vector_count):
        base = vector * rope_dim
        original = values[base : base + rope_dim]
        for idx in range(half_dim):
            angle = position / (theta ** ((2.0 * idx) / rope_dim))
            cos_v = math.cos(angle)
            sin_v = math.sin(angle)
            if interleave:
                even = 2 * idx
                odd = even + 1
                x1 = float(original[even])
                x2 = float(original[odd])
                values[base + even] = x1 * cos_v - x2 * sin_v
                values[base + odd] = x2 * cos_v + x1 * sin_v
            else:
                x1 = float(original[idx])
                x2 = float(original[idx + half_dim])
                values[base + idx] = x1 * cos_v - x2 * sin_v
                values[base + idx + half_dim] = x2 * cos_v + x1 * sin_v


def _read_f32_array_exact(path: Path, *, count: int, label: str) -> array:
    data = path.read_bytes()
    expected_bytes = count * 4
    if len(data) != expected_bytes:
        raise PrefillExecuteError(
            f"{label} bytes {len(data)} do not match expected {expected_bytes}"
        )
    values = array("f")
    values.frombytes(data)
    return values


def _write_singleton_rope_python(
    *,
    q_rope_path: Path,
    k_rope_path: Path,
    q_rope_rotated_path: Path,
    k_rope_rotated_path: Path,
    num_heads: int,
    rope_dim: int,
    position: int,
    theta: float,
    interleave: bool,
) -> None:
    try:
        q_values = _read_f32_array_exact(
            q_rope_path,
            count=num_heads * rope_dim,
            label="singleton q_rope",
        )
        k_values = _read_f32_array_exact(
            k_rope_path,
            count=rope_dim,
            label="singleton k_rope",
        )
        _apply_rope_to_f32_array(
            q_values,
            vector_count=num_heads,
            rope_dim=rope_dim,
            position=position,
            theta=theta,
            interleave=interleave,
        )
        _apply_rope_to_f32_array(
            k_values,
            vector_count=1,
            rope_dim=rope_dim,
            position=position,
            theta=theta,
            interleave=interleave,
        )
        q_rope_rotated_path.write_bytes(q_values.tobytes())
        k_rope_rotated_path.write_bytes(k_values.tobytes())
    except OSError as exc:
        raise PrefillExecuteError(f"failed to run singleton Python RoPE: {exc}") from exc


def _write_singleton_mla_value_from_kv_b(
    *,
    kv_b_output_path: Path,
    output_path: Path,
    num_heads: int,
    qk_nope_dim: int,
    v_head_dim: int,
) -> tuple[int, int]:
    if num_heads <= 0 or qk_nope_dim <= 0 or v_head_dim <= 0:
        raise PrefillExecuteError("singleton MLA value dimensions must be positive")
    head_dim = qk_nope_dim + v_head_dim
    head_bytes = head_dim * 4
    value_bytes = v_head_dim * 4
    skip_bytes = qk_nope_dim * 4
    expected_input_bytes = num_heads * head_bytes
    try:
        actual_input_bytes = kv_b_output_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat singleton kv_b output: {exc}") from exc
    if actual_input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"singleton kv_b output bytes {actual_input_bytes} do not match "
            f"expected {expected_input_bytes}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with kv_b_output_path.open("rb") as source, output_path.open("wb") as out:
            for _head in range(num_heads):
                row = source.read(head_bytes)
                if len(row) != head_bytes:
                    raise PrefillExecuteError("failed to read full singleton kv_b row")
                out.write(row[skip_bytes:])
            trailing = source.read(1)
            if trailing:
                raise PrefillExecuteError("singleton kv_b output has trailing bytes")
    except OSError as exc:
        raise PrefillExecuteError(f"failed to write singleton MLA value: {exc}") from exc
    output_bytes = num_heads * value_bytes
    return output_bytes, head_bytes + value_bytes


def _add_f32_batches_streaming(
    *,
    lhs_path: Path,
    rhs_path: Path,
    output_path: Path,
    batch_tokens: int,
    dim: int,
) -> tuple[int, int]:
    if batch_tokens <= 0 or dim <= 0:
        raise PrefillExecuteError("batch add dimensions must be positive")
    expected_bytes = batch_tokens * dim * 4
    for label, path in (("lhs", lhs_path), ("rhs", rhs_path)):
        try:
            actual = path.stat().st_size
        except OSError as exc:
            raise PrefillExecuteError(f"failed to stat {label} add input {path}: {exc}") from exc
        if actual != expected_bytes:
            raise PrefillExecuteError(
                f"{label} add input bytes {actual} do not match expected {expected_bytes}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_bytes = dim * 4
    try:
        with lhs_path.open("rb") as lhs, rhs_path.open("rb") as rhs, output_path.open(
            "wb"
        ) as out:
            for _ in range(batch_tokens):
                lhs_row = lhs.read(row_bytes)
                rhs_row = rhs.read(row_bytes)
                if len(lhs_row) != row_bytes or len(rhs_row) != row_bytes:
                    raise PrefillExecuteError("failed to read full residual add row")
                lhs_values = struct.unpack(f"<{dim}f", lhs_row)
                rhs_values = struct.unpack(f"<{dim}f", rhs_row)
                out.write(
                    struct.pack(
                        f"<{dim}f",
                        *(left + right for left, right in zip(lhs_values, rhs_values)),
                    )
                )
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stream residual add: {exc}") from exc
    return expected_bytes, 3 * row_bytes


def _silu(value: float) -> float:
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def _swiglu_f32_batches_streaming(
    *,
    gate_path: Path,
    up_path: Path,
    output_path: Path,
    batch_tokens: int,
    dim: int,
) -> tuple[int, int]:
    if batch_tokens <= 0 or dim <= 0:
        raise PrefillExecuteError("SwiGLU dimensions must be positive")
    expected_bytes = batch_tokens * dim * 4
    for label, path in (("gate", gate_path), ("up", up_path)):
        try:
            actual = path.stat().st_size
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to stat {label} SwiGLU input {path}: {exc}"
            ) from exc
        if actual != expected_bytes:
            raise PrefillExecuteError(
                f"{label} SwiGLU input bytes {actual} do not match expected "
                f"{expected_bytes}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_bytes = dim * 4
    try:
        with gate_path.open("rb") as gate, up_path.open("rb") as up, output_path.open(
            "wb"
        ) as out:
            for _ in range(batch_tokens):
                gate_row = gate.read(row_bytes)
                up_row = up.read(row_bytes)
                if len(gate_row) != row_bytes or len(up_row) != row_bytes:
                    raise PrefillExecuteError("failed to read full SwiGLU row")
                gate_values = array("f")
                gate_values.frombytes(gate_row)
                up_values = array("f")
                up_values.frombytes(up_row)
                for idx, gate_value in enumerate(gate_values):
                    gate_values[idx] = _silu(gate_value) * up_values[idx]
                out.write(gate_values.tobytes())
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stream SwiGLU rows: {exc}") from exc
    return expected_bytes, 3 * row_bytes


def run_resident_batch_linear(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    tensor_suffix: str,
    input_f32_path: str | Path,
    output_f32_path: str | Path,
    batch_tokens: int,
    max_resident_matrix_mib: int = 512,
    max_runner_scratch_mib: int = 4096,
    prefill_linear_backend: str = "custom-metal",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    echo_runner_output: bool = True,
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None,
) -> ResidentBatchLinearResult:
    layer = _require_int(layer, "layer")
    batch_tokens = _require_int(batch_tokens, "batch_tokens")
    max_resident_matrix_mib = _positive_integer_mib(
        "max_resident_matrix_mib",
        max_resident_matrix_mib,
    )
    max_runner_scratch_mib = _positive_integer_mib(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )
    prefill_mpsgraph_min_batch_tokens = _require_positive_int(
        prefill_mpsgraph_min_batch_tokens,
        "prefill_mpsgraph_min_batch_tokens",
    )
    prefill_mpsgraph_min_matrix_dim = _require_positive_int(
        prefill_mpsgraph_min_matrix_dim,
        "prefill_mpsgraph_min_matrix_dim",
    )
    if batch_tokens <= 0:
        raise PrefillExecuteError("batch_tokens must be positive")

    runner = Path(runner_path)
    if not runner.exists():
        raise PrefillExecuteError(f"runner not found: {runner}")
    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise PrefillExecuteError("resident layout missing weight_file")
    weight_path = layout_path.parent / weight_file
    if not weight_path.exists():
        raise PrefillExecuteError(f"resident weight file not found: {weight_path}")

    matrix = _find_layer_matrix(layout, layer=layer, suffix=tensor_suffix)
    dtype = str(matrix.get("dtype") or "")
    try:
        mxfp4 = resident_mxfp4_layout_info(layout, matrix)
    except ResidentAffineLayoutError as exc:
        raise PrefillExecuteError(str(exc)) from exc
    if mxfp4 is not None:
        out_dim, in_dim = mxfp4.out_dim, mxfp4.in_dim
        matrix_bytes = mxfp4.total_bytes
        dtype = "mlx-mxfp4"
        for item, label in (
            (mxfp4.weight, "resident MXFP4 weight"),
            (mxfp4.scales, "resident MXFP4 scales"),
        ):
            _check_tensor_backing_span(weight_path, item, label)
    else:
        try:
            affine = resident_affine_int4_layout_info(layout, matrix)
        except ResidentAffineLayoutError as exc:
            raise PrefillExecuteError(str(exc)) from exc
    if mxfp4 is None and affine is not None:
        out_dim, in_dim = affine.out_dim, affine.in_dim
        matrix_bytes = affine.total_bytes
        dtype = "affine-int4"
        for item, label in (
            (affine.weight, "resident affine-int4 weight"),
            (affine.scales, "resident affine-int4 scales"),
            (affine.biases, "resident affine-int4 biases"),
        ):
            _check_tensor_backing_span(weight_path, item, label)
    elif mxfp4 is None:
        out_dim, in_dim = _tensor_shape2(matrix, str(matrix.get("name") or "matrix"))
        matrix_bytes = _int_field(
            matrix,
            "size",
            str(matrix.get("name") or "resident matrix"),
        )
        _check_tensor_backing_span(
            weight_path,
            matrix,
            str(matrix.get("name") or "resident matrix"),
        )
    resolved_backend = _resolve_prefill_linear_backend(
        prefill_linear_backend,
        dtype,
        batch_tokens=batch_tokens,
        in_dim=in_dim,
        out_dim=out_dim,
        mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
    )
    max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
    if matrix_bytes > max_matrix_bytes:
        raise PrefillExecuteError(
            f"resident matrix has {matrix_bytes} bytes, exceeds limit {max_matrix_bytes}"
        )

    input_path = Path(input_f32_path)
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat input f32 file {input_path}: {exc}") from exc
    expected_input_bytes = batch_tokens * in_dim * 4
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"input bytes {input_bytes} do not match batch_tokens*in_dim*f32 "
            f"({expected_input_bytes})"
        )
    output_bytes = batch_tokens * out_dim * 4
    matrix_scratch = _resident_linear_matrix_scratch(
        matrix_bytes=matrix_bytes,
        dtype=dtype,
        in_dim=in_dim,
        out_dim=out_dim,
        backend=resolved_backend,
    )
    estimated_peak = matrix_scratch.matrix_scratch_bytes + input_bytes + output_bytes
    max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    if estimated_peak > max_scratch_bytes:
        raise PrefillExecuteError(
            f"estimated peak {estimated_peak} bytes exceeds scratch limit {max_scratch_bytes}"
        )

    output_path = Path(output_f32_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(runner),
        "--resident-layout",
        str(layout_path),
        "--layer",
        str(layer),
        "--run-resident-linear-batch",
        "--tensor-suffix",
        tensor_suffix,
        "--input-f32",
        str(input_path),
        "--batch-tokens",
        str(batch_tokens),
        "--output-f32",
        str(output_path),
        "--max-resident-matrix-mib",
        str(max_resident_matrix_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    ]
    if resolved_backend != "custom-metal":
        cmd.extend(["--prefill-linear-backend", resolved_backend])
    started = time.perf_counter()
    if resident_linear_server_session is not None:
        runner_stdout = resident_linear_server_session.submit_batch(
            resident_layout_path=layout_path,
            layer=layer,
            tensor_suffix=tensor_suffix,
            input_f32_path=input_path,
            output_f32_path=output_path,
            batch_tokens=batch_tokens,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            prefill_linear_backend=resolved_backend,
        )
        command = (
            str(resident_linear_server_session.runner_path),
            "--run-resident-linear-batch-plan-server-jsonl",
        )
    else:
        completed = _run_command(cmd, echo_output=echo_runner_output)
        runner_stdout = completed.stdout
        command = tuple(cmd)
    elapsed_seconds = time.perf_counter() - started
    runner_timings = _parse_runner_timing_seconds(runner_stdout)
    try:
        actual_output_bytes = output_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat output f32 file {output_path}: {exc}") from exc
    if actual_output_bytes != output_bytes:
        raise PrefillExecuteError(
            f"output bytes {actual_output_bytes} do not match expected {output_bytes}"
        )

    return ResidentBatchLinearResult(
        runner_path=runner,
        resident_layout_path=layout_path,
        input_path=input_path,
        output_path=output_path,
        layer=layer,
        tensor=str(matrix.get("name") or ""),
        tensor_suffix=tensor_suffix,
        dtype=dtype,
        backend=resolved_backend,
        batch_tokens=batch_tokens,
        in_dim=in_dim,
        out_dim=out_dim,
        matrix_bytes=matrix_bytes,
        matrix_scratch_bytes=matrix_scratch.matrix_scratch_bytes,
        matrix_f32_bytes=matrix_scratch.matrix_f32_bytes,
        matrix_raw_conversion_bytes=matrix_scratch.matrix_raw_conversion_bytes,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        estimated_peak_bytes=estimated_peak,
        elapsed_seconds=elapsed_seconds,
        command=command,
        runner_backend_elapsed_seconds=runner_timings.get("backend"),
        runner_matrix_f32_elapsed_seconds=runner_timings.get("matrix_f32"),
        runner_accelerator_elapsed_seconds=runner_timings.get("accelerator"),
    )


def _shared_expert_suffixes(resident_layout_path: str | Path, *, layer: int) -> tuple[str, str, str]:
    return (
        _resolve_layer_matrix_suffix(
            resident_layout_path,
            layer=layer,
            suffixes=(
                ".mlp.shared_experts.gate_proj.weight",
                ".mlp.shared_expert.gate_proj.weight",
                ".shared_experts.gate_proj.weight",
                ".shared_expert.gate_proj.weight",
            ),
        ),
        _resolve_layer_matrix_suffix(
            resident_layout_path,
            layer=layer,
            suffixes=(
                ".mlp.shared_experts.up_proj.weight",
                ".mlp.shared_expert.up_proj.weight",
                ".shared_experts.up_proj.weight",
                ".shared_expert.up_proj.weight",
            ),
        ),
        _resolve_layer_matrix_suffix(
            resident_layout_path,
            layer=layer,
            suffixes=(
                ".mlp.shared_experts.down_proj.weight",
                ".mlp.shared_expert.down_proj.weight",
                ".shared_experts.down_proj.weight",
                ".shared_expert.down_proj.weight",
            ),
        ),
    )


def _shared_expert_mxfp4_metadata(
    layout: dict[str, Any],
    *,
    layer: int,
    gate_suffix: str,
    up_suffix: str,
    down_suffix: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any, Any, Any]:
    gate_matrix = _find_layer_matrix(layout, layer=layer, suffix=gate_suffix)
    up_matrix = _find_layer_matrix(layout, layer=layer, suffix=up_suffix)
    down_matrix = _find_layer_matrix(layout, layer=layer, suffix=down_suffix)
    try:
        gate_mxfp4 = resident_mxfp4_layout_info(layout, gate_matrix)
        up_mxfp4 = resident_mxfp4_layout_info(layout, up_matrix)
        down_mxfp4 = resident_mxfp4_layout_info(layout, down_matrix)
    except ResidentAffineLayoutError as exc:
        raise PrefillExecuteError(str(exc)) from exc
    if gate_mxfp4 is None or up_mxfp4 is None or down_mxfp4 is None:
        raise PrefillExecuteError("shared expert fused batch requires MXFP4 gate/up/down")
    if gate_mxfp4.out_dim != up_mxfp4.out_dim or gate_mxfp4.in_dim != up_mxfp4.in_dim:
        raise PrefillExecuteError("shared expert gate/up MXFP4 shapes do not match")
    if down_mxfp4.out_dim != gate_mxfp4.in_dim or down_mxfp4.in_dim != gate_mxfp4.out_dim:
        raise PrefillExecuteError(
            "shared expert down MXFP4 shape does not match gate/up dimensions"
        )
    if gate_mxfp4.group_size != up_mxfp4.group_size:
        raise PrefillExecuteError("shared expert gate/up MXFP4 group sizes do not match")
    return gate_matrix, up_matrix, down_matrix, gate_mxfp4, up_mxfp4, down_mxfp4


def _shared_expert_fused_candidate_available(
    resident_layout_path: str | Path,
    *,
    layer: int,
) -> bool:
    try:
        layout = _load_json(resident_layout_path)
        gate_suffix, up_suffix, down_suffix = _shared_expert_suffixes(
            resident_layout_path,
            layer=layer,
        )
        _shared_expert_mxfp4_metadata(
            layout,
            layer=layer,
            gate_suffix=gate_suffix,
            up_suffix=up_suffix,
            down_suffix=down_suffix,
        )
    except PrefillExecuteError:
        return False
    return True


def run_resident_shared_expert_batch(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_f32_path: str | Path,
    batch_tokens: int,
    max_resident_matrix_mib: int = 512,
    max_runner_scratch_mib: int = 4096,
    shared_expert_server_session: ResidentSharedExpertBatchServerSession | None = None,
    echo_runner_output: bool = True,
) -> ResidentSharedExpertBatchResult:
    layer = _require_int(layer, "layer")
    batch_tokens = _require_int(batch_tokens, "batch_tokens")
    max_resident_matrix_mib = _positive_integer_mib(
        "max_resident_matrix_mib",
        max_resident_matrix_mib,
    )
    max_runner_scratch_mib = _positive_integer_mib(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )
    if batch_tokens <= 0:
        raise PrefillExecuteError("batch_tokens must be positive")

    runner = Path(runner_path)
    if not runner.exists():
        raise PrefillExecuteError(f"runner not found: {runner}")
    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise PrefillExecuteError("resident layout missing weight_file")
    weight_path = layout_path.parent / weight_file
    if not weight_path.exists():
        raise PrefillExecuteError(f"resident weight file not found: {weight_path}")
    gate_suffix, up_suffix, down_suffix = _shared_expert_suffixes(
        layout_path,
        layer=layer,
    )
    (
        gate_matrix,
        up_matrix,
        down_matrix,
        gate_mxfp4,
        up_mxfp4,
        down_mxfp4,
    ) = _shared_expert_mxfp4_metadata(
        layout,
        layer=layer,
        gate_suffix=gate_suffix,
        up_suffix=up_suffix,
        down_suffix=down_suffix,
    )
    for mxfp4, label in (
        (gate_mxfp4, "shared gate"),
        (up_mxfp4, "shared up"),
        (down_mxfp4, "shared down"),
    ):
        for item, item_label in (
            (mxfp4.weight, f"{label} MXFP4 weight"),
            (mxfp4.scales, f"{label} MXFP4 scales"),
        ):
            _check_tensor_backing_span(weight_path, item, item_label)
    max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
    for matrix_bytes, label in (
        (gate_mxfp4.total_bytes, "shared gate"),
        (up_mxfp4.total_bytes, "shared up"),
        (down_mxfp4.total_bytes, "shared down"),
    ):
        if matrix_bytes > max_matrix_bytes:
            raise PrefillExecuteError(
                f"{label} matrix has {matrix_bytes} bytes, exceeds limit {max_matrix_bytes}"
            )
    hidden_dim = gate_mxfp4.in_dim
    intermediate_dim = gate_mxfp4.out_dim
    input_path = Path(input_f32_path)
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat input f32 file {input_path}: {exc}") from exc
    expected_input_bytes = batch_tokens * hidden_dim * 4
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"input bytes {input_bytes} do not match batch_tokens*hidden_dim*f32 "
            f"({expected_input_bytes})"
        )
    activation_bytes = batch_tokens * intermediate_dim * 4
    output_bytes = batch_tokens * hidden_dim * 4
    matrix_scratch_bytes = (
        _align_up(gate_mxfp4.total_bytes, 2 * 1024 * 1024)
        + _align_up(up_mxfp4.total_bytes, 2 * 1024 * 1024)
        + _align_up(down_mxfp4.total_bytes, 2 * 1024 * 1024)
    )
    matrix_bytes = (
        gate_mxfp4.total_bytes + up_mxfp4.total_bytes + down_mxfp4.total_bytes
    )
    estimated_peak = matrix_scratch_bytes + input_bytes + activation_bytes + output_bytes
    max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    if estimated_peak > max_scratch_bytes:
        raise PrefillExecuteError(
            f"estimated fused shared expert peak {estimated_peak} bytes exceeds "
            f"scratch limit {max_scratch_bytes}"
        )
    output_path = Path(output_f32_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(runner),
        "--resident-layout",
        str(layout_path),
        "--layer",
        str(layer),
        "--run-shared-expert-batch",
        "--input-f32",
        str(input_path),
        "--batch-tokens",
        str(batch_tokens),
        "--output-f32",
        str(output_path),
        "--max-resident-matrix-mib",
        str(max_resident_matrix_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    ]
    started = time.perf_counter()
    if shared_expert_server_session is not None:
        stdout = shared_expert_server_session.submit_batch(
            resident_layout_path=layout_path,
            layer=layer,
            input_f32_path=input_path,
            output_f32_path=output_path,
            batch_tokens=batch_tokens,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
        )
        recorded_command = (
            str(runner),
            "--run-shared-expert-batch-server-jsonl",
        )
    else:
        completed = _run_command(cmd, echo_output=echo_runner_output)
        stdout = completed.stdout
        recorded_command = tuple(cmd)
    elapsed_seconds = time.perf_counter() - started
    runner_timings = _parse_runner_timing_seconds(stdout)
    try:
        actual_output_bytes = output_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat shared expert output {output_path}: {exc}") from exc
    if actual_output_bytes != output_bytes:
        raise PrefillExecuteError(
            f"shared expert output bytes {actual_output_bytes} do not match expected "
            f"{output_bytes}"
        )
    return ResidentSharedExpertBatchResult(
        runner_path=runner,
        resident_layout_path=layout_path,
        input_path=input_path,
        output_path=output_path,
        layer=layer,
        name="shared_expert_batch",
        tensor=f"model.layers.{layer}.mlp.shared_experts",
        dtype="mlx-mxfp4",
        backend="fused-metal",
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        gate_tensor=str(gate_matrix.get("name") or ""),
        up_tensor=str(up_matrix.get("name") or ""),
        down_tensor=str(down_matrix.get("name") or ""),
        gate_matrix_bytes=gate_mxfp4.total_bytes,
        up_matrix_bytes=up_mxfp4.total_bytes,
        down_matrix_bytes=down_mxfp4.total_bytes,
        matrix_bytes=matrix_bytes,
        matrix_scratch_bytes=matrix_scratch_bytes,
        input_bytes=input_bytes,
        activation_bytes=activation_bytes,
        output_bytes=output_bytes,
        estimated_peak_bytes=estimated_peak,
        elapsed_seconds=elapsed_seconds,
        command=recorded_command,
        runner_backend_elapsed_seconds=runner_timings.get("backend"),
    )


def _positive_int_sequence(values: tuple[int, ...] | list[int], label: str) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise PrefillExecuteError(f"{label} must be a non-empty integer list")
    parsed: list[int] = []
    for value in values:
        parsed.append(_require_positive_int(value, label))
    return tuple(sorted(set(parsed)))


def _positive_matrix_shapes(
    values: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    label: str,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise PrefillExecuteError(f"{label} must be a non-empty matrix-shape list")
    parsed: list[tuple[int, int]] = []
    for value in values:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise PrefillExecuteError(f"{label} entries must be (in_dim, out_dim)")
        in_dim = _require_positive_int(value[0], f"{label} in_dim")
        out_dim = _require_positive_int(value[1], f"{label} out_dim")
        parsed.append((in_dim, out_dim))
    return tuple(sorted(set(parsed)))


def _calibration_matrix_dtype(value: object) -> str:
    if not isinstance(value, str):
        raise PrefillExecuteError("matrix_dtype must be F32 or BF16")
    normalized = value.upper()
    if normalized not in CALIBRATION_MATRIX_DTYPE_BYTES:
        raise PrefillExecuteError("matrix_dtype must be F32 or BF16")
    return normalized


def _f32_to_bf16_bytes(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.pack("<H", (rounded >> 16) & 0xFFFF)


def _median_seconds(values: list[float]) -> float:
    if not values:
        raise PrefillExecuteError("elapsed sample list must be non-empty")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _write_resident_linear_calibration_case(
    root: Path,
    *,
    batch_tokens: int,
    in_dim: int,
    out_dim: int,
    matrix_dtype: str,
) -> tuple[Path, Path, str, int, int, int]:
    case_dir = root / f"b{batch_tokens}_k{in_dim}_n{out_dim}"
    resident_dir = case_dir / "resident"
    resident_dir.mkdir(parents=True, exist_ok=True)
    tensor_suffix = f".self_attn.calibration_{in_dim}x{out_dim}.weight"
    tensor_name = f"model.layers.1.self_attn.calibration_{in_dim}x{out_dim}.weight"

    matrix_values = array(
        "f",
        (
            1.0 if row == col else 0.0
            for row in range(out_dim)
            for col in range(in_dim)
        ),
    )
    if matrix_dtype == "F32":
        matrix_payload = matrix_values.tobytes()
    elif matrix_dtype == "BF16":
        matrix_payload = b"".join(_f32_to_bf16_bytes(value) for value in matrix_values)
    else:
        raise PrefillExecuteError("matrix_dtype must be F32 or BF16")
    matrix_bytes = len(matrix_payload)
    (resident_dir / "resident.bin").write_bytes(matrix_payload)
    layout = {
        "version": 1,
        "model_type": "glm_moe_dsa",
        "alignment": 64,
        "weight_file": "resident.bin",
        "total_bytes": matrix_bytes,
        "tensors": [
            {
                "name": tensor_name,
                "offset": 0,
                "size": matrix_bytes,
                "dtype": matrix_dtype,
                "shape": [out_dim, in_dim],
                "category": "calibration",
            }
        ],
    }
    layout_path = resident_dir / "layout.json"
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")

    input_values = array(
        "f",
        (
            float(((token * 17 + dim) % 23) - 11) / 8.0
            for token in range(batch_tokens)
            for dim in range(in_dim)
        ),
    )
    input_bytes = len(input_values) * 4
    input_path = case_dir / "input.f32"
    input_path.write_bytes(input_values.tobytes())
    output_bytes = batch_tokens * out_dim * 4
    return layout_path, input_path, tensor_suffix, matrix_bytes, input_bytes, output_bytes


def _calibration_runtime_policy_flags(
    *,
    batch_tokens: int,
    matrix_dim: int,
) -> dict[str, object]:
    return {
        "source": "prefill_linear_calibration",
        "prefill_mpsgraph_min_batch_tokens": batch_tokens,
        "prefill_mpsgraph_min_matrix_dim": matrix_dim,
        "argv": (
            "--prefill-mpsgraph-min-batch-tokens",
            str(batch_tokens),
            "--prefill-mpsgraph-min-matrix-dim",
            str(matrix_dim),
        ),
    }


def _combined_calibration_runtime_policy_flags(
    *,
    mpsgraph_policy_flags: dict[str, object] | None,
    backend_policy_flags: dict[str, object] | None,
) -> dict[str, object] | None:
    if mpsgraph_policy_flags is None and backend_policy_flags is None:
        return None
    payload: dict[str, object] = {"source": "prefill_linear_calibration"}
    argv: list[str] = []
    backend = (
        backend_policy_flags.get("prefill_linear_backend")
        if isinstance(backend_policy_flags, dict)
        else None
    )
    if isinstance(backend, str) and backend != "auto":
        payload["prefill_linear_backend"] = backend
        argv.extend(["--prefill-linear-backend", backend])
    batch_tokens = (
        mpsgraph_policy_flags.get("prefill_mpsgraph_min_batch_tokens")
        if isinstance(mpsgraph_policy_flags, dict)
        else None
    )
    matrix_dim = (
        mpsgraph_policy_flags.get("prefill_mpsgraph_min_matrix_dim")
        if isinstance(mpsgraph_policy_flags, dict)
        else None
    )
    if isinstance(batch_tokens, int) and isinstance(matrix_dim, int):
        payload["prefill_mpsgraph_min_batch_tokens"] = batch_tokens
        payload["prefill_mpsgraph_min_matrix_dim"] = matrix_dim
        argv.extend(
            [
                "--prefill-mpsgraph-min-batch-tokens",
                str(batch_tokens),
                "--prefill-mpsgraph-min-matrix-dim",
                str(matrix_dim),
            ]
        )
    if not argv:
        return None
    payload["argv"] = tuple(argv)
    return payload


def _calibration_launch_profile(
    policy_flags: dict[str, object] | None,
) -> dict[str, object] | None:
    if policy_flags is None:
        return None
    argv = policy_flags.get("argv")
    return {
        "source": "prefill_linear_calibration",
        "argv_safe_to_replay": True,
        "sections": {"prefill_runtime_policy_flags": policy_flags},
        "argv": tuple(argv) if isinstance(argv, (list, tuple)) else (),
    }


def _recommended_mpsgraph_threshold(
    cases: tuple[ResidentLinearCalibrationCase, ...],
    *,
    min_mpsgraph_speedup: float,
) -> tuple[int, int] | None:
    batch_values = sorted({case.batch_tokens for case in cases})
    dim_values = sorted({case.min_matrix_dim or case.matrix_dim for case in cases})
    best: tuple[int, int, int] | None = None
    for batch_tokens in batch_values:
        for matrix_dim in dim_values:
            included = [
                case
                for case in cases
                if case.batch_tokens >= batch_tokens
                and (case.min_matrix_dim or case.matrix_dim) >= matrix_dim
            ]
            if not included:
                continue
            if any(case.mpsgraph_speedup < min_mpsgraph_speedup for case in included):
                continue
            score = sum(case.estimated_flops for case in included)
            if best is None or score > best[2] or (
                score == best[2] and (batch_tokens, matrix_dim) < (best[0], best[1])
            ):
                best = (batch_tokens, matrix_dim, score)
    if best is None:
        return None
    return best[0], best[1]


def _case_backend_elapsed(
    case: ResidentLinearCalibrationCase,
    backend: str,
) -> float:
    if case.backend_elapsed_seconds is not None and backend in case.backend_elapsed_seconds:
        return float(case.backend_elapsed_seconds[backend])
    if backend == "custom-metal":
        return float(case.custom_elapsed_seconds)
    if backend == "mpsgraph-f32":
        return float(case.mpsgraph_elapsed_seconds)
    if backend == "mps-matrix-f32":
        return float(case.mps_matrix_elapsed_seconds)
    return 0.0


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _calibration_backend_policy_flags(backend: str) -> dict[str, object]:
    return {
        "source": "prefill_linear_calibration",
        "prefill_linear_backend": backend,
        "argv": ("--prefill-linear-backend", backend),
    }


def _calibration_backend_comparison(
    cases: tuple[ResidentLinearCalibrationCase, ...],
    *,
    min_backend_speedup: float,
) -> dict[str, object] | None:
    if not cases:
        return None
    total_flops = sum(case.estimated_flops for case in cases)
    total_elapsed: dict[str, float] = {}
    estimated_tflops: dict[str, float | None] = {}
    speedup_vs_custom: dict[str, float | None] = {}
    min_case_speedup_vs_custom: dict[str, float | None] = {}
    all_cases_meet_min_speedup: dict[str, bool] = {}
    winner_counts = {backend: 0 for backend in PREFILL_LINEAR_CALIBRATION_BACKENDS}
    winner_estimated_flops = {
        backend: 0 for backend in PREFILL_LINEAR_CALIBRATION_BACKENDS
    }

    for case in cases:
        if case.winner in winner_counts:
            winner_counts[case.winner] += 1
            winner_estimated_flops[case.winner] += case.estimated_flops

    custom_total = sum(
        _case_backend_elapsed(case, "custom-metal")
        for case in cases
    )
    for backend in PREFILL_LINEAR_CALIBRATION_BACKENDS:
        backend_total = sum(_case_backend_elapsed(case, backend) for case in cases)
        total_elapsed[backend] = backend_total
        estimated_tflops[backend] = _ratio_or_none(total_flops, backend_total * 1e12)
        speedup_vs_custom[backend] = _ratio_or_none(custom_total, backend_total)
        if backend == "custom-metal":
            min_case_speedup_vs_custom[backend] = 1.0
            all_cases_meet_min_speedup[backend] = True
            continue
        case_speedups = [
            _ratio_or_none(
                _case_backend_elapsed(case, "custom-metal"),
                _case_backend_elapsed(case, backend),
            )
            for case in cases
        ]
        finite_speedups = [
            float(value) for value in case_speedups if value is not None
        ]
        min_speedup = min(finite_speedups) if finite_speedups else None
        min_case_speedup_vs_custom[backend] = min_speedup
        all_cases_meet_min_speedup[backend] = (
            min_speedup is not None and min_speedup >= min_backend_speedup
        )

    accelerated_candidates = [
        backend
        for backend in PREFILL_LINEAR_ACCELERATED_BACKENDS
        if all_cases_meet_min_speedup.get(backend) is True
        and winner_estimated_flops.get(backend, 0) > 0
    ]
    candidates = list(accelerated_candidates)
    if winner_estimated_flops.get("custom-metal", 0) > 0:
        candidates.append("custom-metal")
    recommended = None
    if candidates:
        recommended = min(
            candidates,
            key=lambda backend: (
                total_elapsed[backend],
                -winner_estimated_flops.get(backend, 0),
                backend,
            ),
        )
    policy_flags = (
        _calibration_backend_policy_flags(recommended)
        if recommended is not None
        else None
    )
    return {
        "source": "prefill_linear_calibration",
        "calibrated_backends": PREFILL_LINEAR_CALIBRATION_BACKENDS,
        "case_count": len(cases),
        "total_estimated_flops": total_flops,
        "min_backend_speedup": min_backend_speedup,
        "backend_total_elapsed_seconds": total_elapsed,
        "backend_estimated_tflops": estimated_tflops,
        "backend_speedup_vs_custom": speedup_vs_custom,
        "backend_min_case_speedup_vs_custom": min_case_speedup_vs_custom,
        "backend_all_cases_meet_min_speedup": all_cases_meet_min_speedup,
        "winner_counts": winner_counts,
        "winner_estimated_flops": winner_estimated_flops,
        "recommended_explicit_backend": recommended,
        "recommended_explicit_backend_policy_flags": policy_flags,
    }


def _resident_linear_calibration_work_dir_budget(
    *,
    batch_token_values: tuple[int, ...],
    matrix_shapes: tuple[tuple[int, int], ...],
    repeats: int,
    max_calibration_work_dir_mib: int,
    matrix_dtype: str,
    calibrated_backends: tuple[str, ...] = PREFILL_LINEAR_CALIBRATION_BACKENDS,
) -> ResidentLinearCalibrationWorkDirBudget:
    matrix_bytes = 0
    input_bytes = 0
    single_backend_output_bytes = 0
    dtype_bytes = CALIBRATION_MATRIX_DTYPE_BYTES[matrix_dtype]
    for batch_tokens in batch_token_values:
        for in_dim, out_dim in matrix_shapes:
            matrix_bytes += in_dim * out_dim * dtype_bytes
            input_bytes += batch_tokens * in_dim * 4
            single_backend_output_bytes += batch_tokens * out_dim * 4

    case_count = len(batch_token_values) * len(matrix_shapes)
    backend_count = len(calibrated_backends)
    total_backend_output_bytes = backend_count * repeats * single_backend_output_bytes
    estimated_work_dir_bytes = matrix_bytes + input_bytes + total_backend_output_bytes
    max_work_dir_bytes = max_calibration_work_dir_mib * 1024 * 1024
    return ResidentLinearCalibrationWorkDirBudget(
        source="prefill_linear_calibration",
        case_count=case_count,
        batch_token_count=len(batch_token_values),
        shape_count=len(matrix_shapes),
        repeats=repeats,
        backend_count=backend_count,
        calibrated_backends=calibrated_backends,
        backend_output_file_count=backend_count * repeats * case_count,
        matrix_bytes=matrix_bytes,
        input_bytes=input_bytes,
        single_backend_output_bytes=single_backend_output_bytes,
        total_backend_output_bytes=total_backend_output_bytes,
        single_case_bytes=matrix_bytes + input_bytes + single_backend_output_bytes,
        estimated_work_dir_bytes=estimated_work_dir_bytes,
        max_calibration_work_dir_bytes=max_work_dir_bytes,
        max_calibration_work_dir_mib=max_calibration_work_dir_mib,
    )


def _nearest_existing_path(path: Path) -> Path:
    probe = path if path.exists() else path.parent
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            return probe
        probe = parent
    return probe


def _check_resident_linear_calibration_work_dir_disk(
    budget: ResidentLinearCalibrationWorkDirBudget,
    *,
    path: Path,
    free_margin_mib: int,
) -> ResidentLinearCalibrationWorkDirBudget:
    margin_bytes = free_margin_mib * 1024 * 1024
    usage_path = _nearest_existing_path(path)
    available_bytes = int(shutil.disk_usage(usage_path).free)
    required_bytes = budget.estimated_work_dir_bytes + margin_bytes
    checked = replace(
        budget,
        disk_usage_path=usage_path,
        disk_available_bytes=available_bytes,
        disk_required_bytes=required_bytes,
        disk_safety_margin_bytes=margin_bytes,
    )
    if available_bytes < required_bytes:
        raise PrefillExecuteError(
            "calibration work dir needs "
            f"{budget.estimated_work_dir_bytes} bytes plus {margin_bytes} bytes "
            f"free margin on {usage_path}, only {available_bytes} bytes available"
        )
    return checked


def _run_resident_linear_calibration_in_dir(
    root: Path,
    *,
    runner_path: str | Path,
    batch_token_values: tuple[int, ...],
    matrix_dim_values: tuple[int, ...],
    matrix_shapes: tuple[tuple[int, int], ...],
    repeats: int,
    min_mpsgraph_speedup: float,
    max_calibration_case_bytes: int,
    max_resident_matrix_mib: int,
    max_runner_scratch_mib: int,
    work_dir_budget: ResidentLinearCalibrationWorkDirBudget,
    matrix_dtype: str,
    echo_runner_output: bool,
    kept_work_dir: bool,
) -> ResidentLinearCalibrationResult:
    runner = Path(runner_path)
    cases: list[ResidentLinearCalibrationCase] = []
    for batch_tokens in batch_token_values:
        for in_dim, out_dim in matrix_shapes:
            min_dim = min(in_dim, out_dim)
            layout_path, input_path, tensor_suffix, matrix_bytes, input_bytes, output_bytes = (
                _write_resident_linear_calibration_case(
                    root,
                    batch_tokens=batch_tokens,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    matrix_dtype=matrix_dtype,
                )
            )
            case_bytes = matrix_bytes + input_bytes + output_bytes
            if case_bytes > max_calibration_case_bytes:
                raise PrefillExecuteError(
                    f"calibration case b={batch_tokens} shape={in_dim}x{out_dim} needs "
                    f"{case_bytes} bytes, exceeds limit {max_calibration_case_bytes}"
                )
            times_by_backend: dict[str, list[float]] = {
                backend: [] for backend in PREFILL_LINEAR_CALIBRATION_BACKENDS
            }
            peak_by_backend: dict[str, int] = {
                backend: 0 for backend in PREFILL_LINEAR_CALIBRATION_BACKENDS
            }
            case_dir = root / f"b{batch_tokens}_k{in_dim}_n{out_dim}"
            for repeat in range(repeats):
                for backend in PREFILL_LINEAR_CALIBRATION_BACKENDS:
                    result = run_resident_batch_linear(
                        runner_path=runner,
                        resident_layout_path=layout_path,
                        layer=1,
                        tensor_suffix=tensor_suffix,
                        input_f32_path=input_path,
                        output_f32_path=(
                            case_dir
                            / f"{backend.replace('-', '_')}_{repeat}.f32"
                        ),
                        batch_tokens=batch_tokens,
                        max_resident_matrix_mib=max_resident_matrix_mib,
                        max_runner_scratch_mib=max_runner_scratch_mib,
                        prefill_linear_backend=backend,
                        echo_runner_output=echo_runner_output,
                    )
                    times_by_backend[backend].append(result.elapsed_seconds)
                    peak_by_backend[backend] = result.estimated_peak_bytes
            elapsed_by_backend = {
                backend: _median_seconds(times)
                for backend, times in times_by_backend.items()
            }
            custom_elapsed = elapsed_by_backend["custom-metal"]
            mpsgraph_elapsed = elapsed_by_backend["mpsgraph-f32"]
            mps_matrix_elapsed = elapsed_by_backend["mps-matrix-f32"]
            speedup = (
                float("inf")
                if mpsgraph_elapsed <= 0.0
                else custom_elapsed / mpsgraph_elapsed
            )
            mps_matrix_speedup = (
                float("inf")
                if mps_matrix_elapsed <= 0.0
                else custom_elapsed / mps_matrix_elapsed
            )
            winner = min(
                elapsed_by_backend,
                key=lambda backend: (elapsed_by_backend[backend], backend),
            )
            cases.append(
                ResidentLinearCalibrationCase(
                    batch_tokens=batch_tokens,
                    matrix_dim=min_dim,
                    matrix_bytes=matrix_bytes,
                    input_bytes=input_bytes,
                    output_bytes=output_bytes,
                    estimated_flops=2 * batch_tokens * in_dim * out_dim,
                    custom_elapsed_seconds=custom_elapsed,
                    mpsgraph_elapsed_seconds=mpsgraph_elapsed,
                    mpsgraph_speedup=speedup,
                    mpsgraph_meets_threshold=speedup >= min_mpsgraph_speedup,
                    winner=winner,
                    custom_estimated_peak_bytes=peak_by_backend["custom-metal"],
                    mpsgraph_estimated_peak_bytes=peak_by_backend["mpsgraph-f32"],
                    in_dim=in_dim,
                    out_dim=out_dim,
                    min_matrix_dim=min_dim,
                    mps_matrix_elapsed_seconds=mps_matrix_elapsed,
                    mps_matrix_speedup=mps_matrix_speedup,
                    mps_matrix_estimated_peak_bytes=(
                        peak_by_backend["mps-matrix-f32"]
                    ),
                    backend_elapsed_seconds=elapsed_by_backend,
                    backend_estimated_peak_bytes=peak_by_backend,
                    matrix_dtype=matrix_dtype,
                )
            )

    case_tuple = tuple(cases)
    recommendation = _recommended_mpsgraph_threshold(
        case_tuple,
        min_mpsgraph_speedup=min_mpsgraph_speedup,
    )
    backend_comparison = _calibration_backend_comparison(
        case_tuple,
        min_backend_speedup=min_mpsgraph_speedup,
    )
    mpsgraph_policy_flags = (
        _calibration_runtime_policy_flags(
            batch_tokens=recommendation[0],
            matrix_dim=recommendation[1],
        )
        if recommendation is not None
        else None
    )
    backend_policy_flags = None
    if isinstance(backend_comparison, dict):
        candidate_policy = backend_comparison.get(
            "recommended_explicit_backend_policy_flags"
        )
        if isinstance(candidate_policy, dict):
            candidate_backend = candidate_policy.get("prefill_linear_backend")
            if candidate_backend != "custom-metal" or mpsgraph_policy_flags is None:
                backend_policy_flags = candidate_policy
    policy_flags = _combined_calibration_runtime_policy_flags(
        mpsgraph_policy_flags=mpsgraph_policy_flags,
        backend_policy_flags=backend_policy_flags,
    )
    return ResidentLinearCalibrationResult(
        runner_path=runner,
        work_dir=root,
        kept_work_dir=kept_work_dir,
        batch_token_values=batch_token_values,
        matrix_dim_values=matrix_dim_values,
        repeats=repeats,
        min_mpsgraph_speedup=min_mpsgraph_speedup,
        max_calibration_case_bytes=max_calibration_case_bytes,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        recommended_prefill_mpsgraph_min_batch_tokens=(
            recommendation[0] if recommendation is not None else None
        ),
        recommended_prefill_mpsgraph_min_matrix_dim=(
            recommendation[1] if recommendation is not None else None
        ),
        suggested_prefill_runtime_policy_flags=policy_flags,
        suggested_launch_profile=_calibration_launch_profile(policy_flags),
        cases=case_tuple,
        matrix_shapes=matrix_shapes,
        work_dir_budget=work_dir_budget,
        backend_comparison=backend_comparison,
        matrix_dtype=matrix_dtype,
    )


def run_resident_linear_calibration(
    *,
    runner_path: str | Path,
    batch_token_values: tuple[int, ...] | list[int] = (32, 64, 128),
    matrix_dim_values: tuple[int, ...] | list[int] = (16, 32, 64),
    matrix_shapes: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
    repeats: int = 1,
    min_mpsgraph_speedup: float = 1.0,
    max_calibration_case_mib: int = 64,
    max_resident_matrix_mib: int = 64,
    max_runner_scratch_mib: int = 64,
    max_calibration_work_dir_mib: int = 8192,
    calibration_work_dir_free_margin_mib: int = 512,
    matrix_dtype: str = "F32",
    work_dir: str | Path | None = None,
    keep_work_dir: bool = False,
    echo_runner_output: bool = False,
) -> ResidentLinearCalibrationResult:
    batch_tokens = _positive_int_sequence(batch_token_values, "batch_token_values")
    if matrix_shapes is None:
        matrix_dims = _positive_int_sequence(matrix_dim_values, "matrix_dim_values")
        resolved_matrix_shapes = tuple((dim, dim) for dim in matrix_dims)
    else:
        resolved_matrix_shapes = _positive_matrix_shapes(matrix_shapes, "matrix_shapes")
        matrix_dims = tuple(
            sorted({min(in_dim, out_dim) for in_dim, out_dim in resolved_matrix_shapes})
        )
    matrix_dtype = _calibration_matrix_dtype(matrix_dtype)
    repeats = _require_positive_int(repeats, "repeats")
    if repeats > 10:
        raise PrefillExecuteError("repeats must be <= 10")
    min_mpsgraph_speedup = _require_positive_number(
        min_mpsgraph_speedup,
        "min_mpsgraph_speedup",
    )
    max_case_mib = _positive_integer_mib(
        "max_calibration_case_mib",
        max_calibration_case_mib,
    )
    max_resident_matrix_mib = _positive_integer_mib(
        "max_resident_matrix_mib",
        max_resident_matrix_mib,
    )
    max_runner_scratch_mib = _positive_integer_mib(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )
    max_calibration_work_dir_mib = _positive_integer_mib(
        "max_calibration_work_dir_mib",
        max_calibration_work_dir_mib,
    )
    calibration_work_dir_free_margin_mib = _positive_integer_mib(
        "calibration_work_dir_free_margin_mib",
        calibration_work_dir_free_margin_mib,
    )
    max_case_bytes = max_case_mib * 1024 * 1024
    work_dir_budget = _resident_linear_calibration_work_dir_budget(
        batch_token_values=batch_tokens,
        matrix_shapes=resolved_matrix_shapes,
        repeats=repeats,
        max_calibration_work_dir_mib=max_calibration_work_dir_mib,
        matrix_dtype=matrix_dtype,
    )
    if (
        work_dir_budget.estimated_work_dir_bytes
        > work_dir_budget.max_calibration_work_dir_bytes
    ):
        raise PrefillExecuteError(
            "calibration work dir needs "
            f"{work_dir_budget.estimated_work_dir_bytes} bytes, exceeds "
            "max_calibration_work_dir_mib "
            f"{max_calibration_work_dir_mib} MiB"
        )
    disk_probe_path = Path(work_dir) if work_dir is not None else Path("/private/tmp")
    work_dir_budget = _check_resident_linear_calibration_work_dir_disk(
        work_dir_budget,
        path=disk_probe_path,
        free_margin_mib=calibration_work_dir_free_margin_mib,
    )

    if work_dir is not None:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        return _run_resident_linear_calibration_in_dir(
            root,
            runner_path=runner_path,
            batch_token_values=batch_tokens,
            matrix_dim_values=matrix_dims,
            matrix_shapes=resolved_matrix_shapes,
            repeats=repeats,
            min_mpsgraph_speedup=min_mpsgraph_speedup,
            max_calibration_case_bytes=max_case_bytes,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            work_dir_budget=work_dir_budget,
            matrix_dtype=matrix_dtype,
            echo_runner_output=echo_runner_output,
            kept_work_dir=True,
        )

    if keep_work_dir:
        root = Path(
            tempfile.mkdtemp(prefix="largerlm-prefill-linear-calibration-", dir="/private/tmp")
        )
        return _run_resident_linear_calibration_in_dir(
            root,
            runner_path=runner_path,
            batch_token_values=batch_tokens,
            matrix_dim_values=matrix_dims,
            matrix_shapes=resolved_matrix_shapes,
            repeats=repeats,
            min_mpsgraph_speedup=min_mpsgraph_speedup,
            max_calibration_case_bytes=max_case_bytes,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            work_dir_budget=work_dir_budget,
            matrix_dtype=matrix_dtype,
            echo_runner_output=echo_runner_output,
            kept_work_dir=True,
        )

    with tempfile.TemporaryDirectory(
        prefix="largerlm-prefill-linear-calibration-",
        dir="/private/tmp",
    ) as temporary:
        return _run_resident_linear_calibration_in_dir(
            Path(temporary),
            runner_path=runner_path,
            batch_token_values=batch_tokens,
            matrix_dim_values=matrix_dims,
            matrix_shapes=resolved_matrix_shapes,
            repeats=repeats,
            min_mpsgraph_speedup=min_mpsgraph_speedup,
            max_calibration_case_bytes=max_case_bytes,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            work_dir_budget=work_dir_budget,
            matrix_dtype=matrix_dtype,
            echo_runner_output=echo_runner_output,
            kept_work_dir=False,
        )


def run_resident_batch_rmsnorm(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    norm_suffix: str,
    input_f32_path: str | Path,
    output_f32_path: str | Path,
    batch_tokens: int,
    rms_norm_eps: float = 1e-5,
    max_runner_scratch_mib: int = 4096,
    echo_runner_output: bool = True,
    rmsnorm_server_session: ResidentBatchRMSNormServerSession | None = None,
) -> ResidentBatchRMSNormResult:
    layer = _require_int(layer, "layer")
    batch_tokens = _require_int(batch_tokens, "batch_tokens")
    max_runner_scratch_mib = _positive_integer_mib(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )
    rms_norm_eps = _require_number(rms_norm_eps, "rms_norm_eps")
    if batch_tokens <= 0:
        raise PrefillExecuteError("batch_tokens must be positive")
    if rms_norm_eps < 0:
        raise PrefillExecuteError("rms_norm_eps must be non-negative")

    runner = Path(runner_path)
    if not runner.exists():
        raise PrefillExecuteError(f"runner not found: {runner}")
    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise PrefillExecuteError("resident layout missing weight_file")
    weight_path = layout_path.parent / weight_file
    if not weight_path.exists():
        raise PrefillExecuteError(f"resident weight file not found: {weight_path}")

    vector = _find_layer_vector(layout, layer=layer, suffix=norm_suffix)
    hidden_dim = _tensor_shape1(vector, str(vector.get("name") or "vector"))
    dtype = str(vector.get("dtype") or "")
    _check_tensor_backing_span(
        weight_path,
        vector,
        str(vector.get("name") or "resident vector"),
    )
    vector_bytes = _int_field(
        vector,
        "size",
        str(vector.get("name") or "resident vector"),
    )

    input_path = Path(input_f32_path)
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat input f32 file {input_path}: {exc}") from exc
    expected_input_bytes = batch_tokens * hidden_dim * 4
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"input bytes {input_bytes} do not match batch_tokens*hidden_dim*f32 "
            f"({expected_input_bytes})"
        )
    output_bytes = expected_input_bytes
    weight_f32_bytes = hidden_dim * 4
    estimated_peak = input_bytes + output_bytes + vector_bytes + weight_f32_bytes
    max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    if estimated_peak > max_scratch_bytes:
        raise PrefillExecuteError(
            f"estimated peak {estimated_peak} bytes exceeds scratch limit {max_scratch_bytes}"
        )

    output_path = Path(output_f32_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(runner),
        "--resident-layout",
        str(layout_path),
        "--layer",
        str(layer),
        "--run-rmsnorm-batch",
        "--norm-suffix",
        norm_suffix,
        "--input-f32",
        str(input_path),
        "--batch-tokens",
        str(batch_tokens),
        "--rms-norm-eps",
        f"{rms_norm_eps:.9g}",
        "--output-f32",
        str(output_path),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    ]
    if rmsnorm_server_session is not None:
        rmsnorm_server_session.submit_batch(
            resident_layout_path=layout_path,
            layer=layer,
            norm_suffix=norm_suffix,
            input_f32_path=input_path,
            output_f32_path=output_path,
            batch_tokens=batch_tokens,
            rms_norm_eps=rms_norm_eps,
            max_runner_scratch_mib=max_runner_scratch_mib,
        )
        command = (
            str(rmsnorm_server_session.runner_path),
            "--run-rmsnorm-batch-server-jsonl",
        )
    else:
        _run_command(cmd, echo_output=echo_runner_output)
        command = tuple(cmd)
    try:
        actual_output_bytes = output_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat output f32 file {output_path}: {exc}") from exc
    if actual_output_bytes != output_bytes:
        raise PrefillExecuteError(
            f"output bytes {actual_output_bytes} do not match expected {output_bytes}"
        )

    return ResidentBatchRMSNormResult(
        runner_path=runner,
        resident_layout_path=layout_path,
        input_path=input_path,
        output_path=output_path,
        layer=layer,
        tensor=str(vector.get("name") or ""),
        norm_suffix=norm_suffix,
        dtype=dtype,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        vector_bytes=vector_bytes,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        estimated_peak_bytes=estimated_peak,
        rms_norm_eps=rms_norm_eps,
        command=command,
    )


def run_prefill_attention_prefix_batch(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_dir: str | Path,
    batch_tokens: int,
    norm_suffix: str = ".input_layernorm.weight",
    q_a_suffix: str = ".self_attn.q_a_proj.weight",
    kv_a_suffix: str = ".self_attn.kv_a_proj_with_mqa.weight",
    rms_norm_eps: float = 1e-5,
    max_resident_matrix_mib: int = 512,
    max_runner_scratch_mib: int = 4096,
    prefill_linear_backend: str = "custom-metal",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    echo_runner_output: bool = True,
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None,
    rmsnorm_server_session: ResidentBatchRMSNormServerSession | None = None,
) -> PrefillAttentionPrefixResult:
    """Run the bounded GLM attention prefill prefix for one layer and token batch."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_output = out_dir / "input_layernorm.f32"
    q_a_output = out_dir / "q_a_proj.f32"
    kv_a_output = out_dir / "kv_a_proj_with_mqa.f32"

    norm_result = run_resident_batch_rmsnorm(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        norm_suffix=norm_suffix,
        input_f32_path=input_f32_path,
        output_f32_path=norm_output,
        batch_tokens=batch_tokens,
        rms_norm_eps=rms_norm_eps,
        max_runner_scratch_mib=max_runner_scratch_mib,
        echo_runner_output=echo_runner_output,
        rmsnorm_server_session=rmsnorm_server_session,
    )
    q_a_result = run_resident_batch_linear(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        tensor_suffix=q_a_suffix,
        input_f32_path=norm_output,
        output_f32_path=q_a_output,
        batch_tokens=batch_tokens,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
    )
    kv_a_result = run_resident_batch_linear(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        tensor_suffix=kv_a_suffix,
        input_f32_path=norm_output,
        output_f32_path=kv_a_output,
        batch_tokens=batch_tokens,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
    )

    estimated_peak = max(
        norm_result.estimated_peak_bytes,
        q_a_result.estimated_peak_bytes,
        kv_a_result.estimated_peak_bytes,
    )
    return PrefillAttentionPrefixResult(
        runner_path=norm_result.runner_path,
        resident_layout_path=norm_result.resident_layout_path,
        input_path=norm_result.input_path,
        output_dir=out_dir,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=norm_result.hidden_dim,
        q_a_dim=q_a_result.out_dim,
        kv_a_dim=kv_a_result.out_dim,
        input_bytes=norm_result.input_bytes,
        norm_output_bytes=norm_result.output_bytes,
        q_a_output_bytes=q_a_result.output_bytes,
        kv_a_output_bytes=kv_a_result.output_bytes,
        estimated_peak_bytes=estimated_peak,
        input_layernorm=norm_result,
        q_a_proj=q_a_result,
        kv_a_proj_with_mqa=kv_a_result,
    )


def _resident_matrix_metadata(
    *,
    layout: dict[str, Any],
    weight_path: Path,
    matrix: dict[str, Any],
    label: str,
) -> tuple[int, int, int, str]:
    dtype = str(matrix.get("dtype") or "")
    try:
        mxfp4 = resident_mxfp4_layout_info(layout, matrix)
    except ResidentAffineLayoutError as exc:
        raise PrefillExecuteError(str(exc)) from exc
    if mxfp4 is not None:
        for item, item_label in (
            (mxfp4.weight, f"{label} MXFP4 weight"),
            (mxfp4.scales, f"{label} MXFP4 scales"),
        ):
            _check_tensor_backing_span(weight_path, item, item_label)
        return mxfp4.out_dim, mxfp4.in_dim, mxfp4.total_bytes, "mlx-mxfp4"

    try:
        affine = resident_affine_int4_layout_info(layout, matrix)
    except ResidentAffineLayoutError as exc:
        raise PrefillExecuteError(str(exc)) from exc
    if affine is not None:
        for item, item_label in (
            (affine.weight, f"{label} affine-int4 weight"),
            (affine.scales, f"{label} affine-int4 scales"),
            (affine.biases, f"{label} affine-int4 biases"),
        ):
            _check_tensor_backing_span(weight_path, item, item_label)
        return affine.out_dim, affine.in_dim, affine.total_bytes, "affine-int4"

    out_dim, in_dim = _tensor_shape2(matrix, label)
    matrix_bytes = _int_field(matrix, "size", label)
    _check_tensor_backing_span(weight_path, matrix, label)
    return out_dim, in_dim, matrix_bytes, dtype


def _resident_linear_result_metadata(
    *,
    runner: Path,
    layout_path: Path,
    matrix: dict[str, Any],
    tensor_suffix: str,
    input_path: Path,
    output_path: Path,
    layer: int,
    batch_tokens: int,
    in_dim: int,
    out_dim: int,
    matrix_bytes: int,
    dtype: str,
    backend: str,
    command: tuple[str, ...],
) -> ResidentBatchLinearResult:
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat fused input {input_path}: {exc}") from exc
    output_bytes = batch_tokens * out_dim * 4
    matrix_scratch = _resident_linear_matrix_scratch(
        matrix_bytes=matrix_bytes,
        dtype=dtype,
        in_dim=in_dim,
        out_dim=out_dim,
        backend=backend,
    )
    return ResidentBatchLinearResult(
        runner_path=runner,
        resident_layout_path=layout_path,
        input_path=input_path,
        output_path=output_path,
        layer=layer,
        tensor=str(matrix.get("name") or ""),
        tensor_suffix=tensor_suffix,
        dtype=dtype,
        backend=backend,
        batch_tokens=batch_tokens,
        in_dim=in_dim,
        out_dim=out_dim,
        matrix_bytes=matrix_bytes,
        matrix_scratch_bytes=matrix_scratch.matrix_scratch_bytes,
        matrix_f32_bytes=matrix_scratch.matrix_f32_bytes,
        matrix_raw_conversion_bytes=matrix_scratch.matrix_raw_conversion_bytes,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        estimated_peak_bytes=matrix_scratch.matrix_scratch_bytes
        + input_bytes
        + output_bytes,
        elapsed_seconds=0.0,
        command=command,
    )


def _resident_rmsnorm_result_metadata(
    *,
    runner: Path,
    layout_path: Path,
    vector: dict[str, Any],
    norm_suffix: str,
    input_path: Path,
    output_path: Path,
    layer: int,
    batch_tokens: int,
    hidden_dim: int,
    rms_norm_eps: float,
    command: tuple[str, ...],
) -> ResidentBatchRMSNormResult:
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat fused norm input {input_path}: {exc}") from exc
    vector_bytes = _int_field(vector, "size", str(vector.get("name") or "norm vector"))
    output_bytes = batch_tokens * hidden_dim * 4
    return ResidentBatchRMSNormResult(
        runner_path=runner,
        resident_layout_path=layout_path,
        input_path=input_path,
        output_path=output_path,
        layer=layer,
        tensor=str(vector.get("name") or ""),
        norm_suffix=norm_suffix,
        dtype=str(vector.get("dtype") or ""),
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        vector_bytes=vector_bytes,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        estimated_peak_bytes=input_bytes + output_bytes + vector_bytes,
        rms_norm_eps=rms_norm_eps,
        command=command,
    )


def _stat_expected_file(path: Path, expected_bytes: int, label: str) -> None:
    try:
        actual_bytes = path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat {label} {path}: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise PrefillExecuteError(
            f"{label} bytes {actual_bytes} do not match expected {expected_bytes}"
        )


def _visible_dsa_topk_rows(
    *,
    start_position: int,
    batch_tokens: int,
    context_length: int,
    index_topk: int,
) -> tuple[tuple[int, ...], ...] | None:
    if index_topk <= 0:
        raise PrefillExecuteError("DSA index_topk must be positive")
    if start_position < 0 or batch_tokens <= 0 or context_length <= 0:
        raise PrefillExecuteError("DSA visible top-k positions are invalid")
    if start_position + batch_tokens > context_length:
        raise PrefillExecuteError("DSA visible top-k positions exceed context_length")
    rows: list[tuple[int, ...]] = []
    for token_offset in range(batch_tokens):
        position = start_position + token_offset
        visible_tokens = min(context_length, position + 1)
        if visible_tokens > index_topk:
            return None
        rows.append(tuple(range(visible_tokens)))
    return tuple(rows)


def _write_visible_dsa_topk_u32(
    path: Path,
    *,
    rows: tuple[tuple[int, ...], ...],
    index_topk: int,
) -> int:
    if index_topk <= 0:
        raise PrefillExecuteError("DSA index_topk must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            for row in rows:
                if len(row) > index_topk:
                    raise PrefillExecuteError("DSA visible top-k row exceeds index_topk")
                values = [len(row), *row, *([0] * (index_topk - len(row)))]
                handle.write(struct.pack(f"<{len(values)}I", *values))
        tmp_path.replace(path)
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise PrefillExecuteError(f"failed to write visible DSA indices: {exc}") from exc
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return len(rows) * (index_topk + 1) * 4

def _run_prefill_attention_projection_batch_fused_runner(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_dir: str | Path,
    batch_tokens: int,
    norm_suffix: str,
    q_a_suffix: str,
    q_a_norm_suffix: str,
    q_b_suffix: str,
    kv_a_suffix: str,
    kv_a_norm_suffix: str,
    kv_b_suffix: str,
    rms_norm_eps: float,
    max_resident_matrix_mib: int,
    max_runner_scratch_mib: int,
    echo_runner_output: bool,
    attention_projection_server_session: AttentionProjectionsServerSession | None = None,
) -> PrefillAttentionProjectionBatchResult:
    runner = Path(runner_path)
    if not runner.exists():
        raise PrefillExecuteError(f"runner not found: {runner}")
    batch_tokens = _require_int(batch_tokens, "batch_tokens")
    if batch_tokens <= 0:
        raise PrefillExecuteError("batch_tokens must be positive")
    layout_path = Path(resident_layout_path)
    layout = _load_json(layout_path)
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise PrefillExecuteError("resident layout missing weight_file")
    weight_path = layout_path.parent / weight_file
    if not weight_path.exists():
        raise PrefillExecuteError(f"resident weight file not found: {weight_path}")

    input_path = Path(input_f32_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_resident_matrix_mib = _positive_integer_mib(
        "max_resident_matrix_mib",
        max_resident_matrix_mib,
    )
    max_runner_scratch_mib = _positive_integer_mib(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )

    input_norm_output = out_dir / "attn_input_norm.f32"
    q_a_output = out_dir / "attn_q_a.f32"
    q_a_norm_output = out_dir / "attn_q_a_norm.f32"
    q_b_output = out_dir / "attn_q_b.f32"
    kv_a_output = out_dir / "attn_kv_a.f32"
    kv_a_norm_output = out_dir / "attn_kv_a_norm.f32"
    kv_b_output = out_dir / "attn_kv_b.f32"
    kv_a_lora_path = out_dir / "kv_a_lora.f32"
    kv_a_rope_path = out_dir / "kv_a_rope.f32"

    q_a_tensor = _find_layer_matrix(layout, layer=layer, suffix=q_a_suffix)
    q_b_tensor = _find_layer_matrix(layout, layer=layer, suffix=q_b_suffix)
    kv_a_tensor = _find_layer_matrix(layout, layer=layer, suffix=kv_a_suffix)
    kv_b_tensor = _find_layer_tensor_optional(layout, layer=layer, suffix=kv_b_suffix)
    input_norm_tensor = _find_layer_vector(layout, layer=layer, suffix=norm_suffix)
    q_a_norm_tensor = _find_layer_vector(layout, layer=layer, suffix=q_a_norm_suffix)
    kv_a_norm_tensor = _find_layer_vector(layout, layer=layer, suffix=kv_a_norm_suffix)

    q_a_dim, hidden_dim, q_a_bytes, q_a_dtype = _resident_matrix_metadata(
        layout=layout,
        weight_path=weight_path,
        matrix=q_a_tensor,
        label=str(q_a_tensor.get("name") or "attention q_a"),
    )
    q_b_dim, q_b_in_dim, q_b_bytes, q_b_dtype = _resident_matrix_metadata(
        layout=layout,
        weight_path=weight_path,
        matrix=q_b_tensor,
        label=str(q_b_tensor.get("name") or "attention q_b"),
    )
    kv_a_dim, kv_a_in_dim, kv_a_bytes, kv_a_dtype = _resident_matrix_metadata(
        layout=layout,
        weight_path=weight_path,
        matrix=kv_a_tensor,
        label=str(kv_a_tensor.get("name") or "attention kv_a"),
    )
    if q_b_in_dim != q_a_dim or kv_a_in_dim != hidden_dim:
        raise PrefillExecuteError("fused attention projection dimensions are inconsistent")
    if _tensor_shape1(input_norm_tensor, str(input_norm_tensor.get("name") or "input norm")) != hidden_dim:
        raise PrefillExecuteError("input norm dim mismatch")
    if _tensor_shape1(q_a_norm_tensor, str(q_a_norm_tensor.get("name") or "q_a norm")) != q_a_dim:
        raise PrefillExecuteError("q_a norm dim mismatch")
    kv_lora_dim = _tensor_shape1(
        kv_a_norm_tensor,
        str(kv_a_norm_tensor.get("name") or "kv_a norm"),
    )
    kv_rope_dim = kv_a_dim - kv_lora_dim
    if kv_rope_dim < 0:
        raise PrefillExecuteError("kv_a output dim is smaller than kv_lora_dim")

    max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
    for label, bytes_value in (
        ("attention q_a", q_a_bytes),
        ("attention q_b", q_b_bytes),
        ("attention kv_a", kv_a_bytes),
    ):
        if bytes_value > max_matrix_bytes:
            raise PrefillExecuteError(f"{label} has {bytes_value} bytes, exceeds limit {max_matrix_bytes}")

    attention_value_source = "kv_b_proj"
    kv_b_dim = 0
    kv_b_output_bytes = 0
    kv_b_peak = 0
    kv_b_result: ResidentBatchLinearResult | None = None
    if kv_b_tensor is not None:
        kv_b_dim, kv_b_in_dim, kv_b_bytes, kv_b_dtype = _resident_matrix_metadata(
            layout=layout,
            weight_path=weight_path,
            matrix=kv_b_tensor,
            label=str(kv_b_tensor.get("name") or "attention kv_b"),
        )
        if kv_b_in_dim != kv_lora_dim:
            raise PrefillExecuteError("kv_b input dim mismatch")
        if kv_b_bytes > max_matrix_bytes:
            raise PrefillExecuteError(
                f"attention kv_b has {kv_b_bytes} bytes, exceeds limit {max_matrix_bytes}"
            )
        kv_b_output_bytes = batch_tokens * kv_b_dim * 4
    else:
        embed_q = _find_layer_matrix(
            layout,
            layer=layer,
            suffix=".self_attn.embed_q.weight",
        )
        unembed_out = _find_layer_matrix(
            layout,
            layer=layer,
            suffix=".self_attn.unembed_out.weight",
        )
        (
            embed_heads,
            embed_kv_lora,
            embed_qk_nope,
            embed_storage_bytes,
            embed_f32_bytes,
        ) = _tensor_shape3_value_source(
            layout,
            embed_q,
            "embed_q.weight",
            weight_path,
        )
        (
            unembed_heads,
            unembed_v_head,
            unembed_kv_lora,
            unembed_storage_bytes,
            unembed_f32_bytes,
        ) = _tensor_shape3_value_source(
            layout,
            unembed_out,
            "unembed_out.weight",
            weight_path,
        )
        if (
            embed_kv_lora != kv_lora_dim
            or unembed_kv_lora != kv_lora_dim
            or embed_heads != unembed_heads
        ):
            raise PrefillExecuteError("absorbed attention alias dims are inconsistent")
        attention_value_source = "absorbed-alias"
        kv_b_dim = embed_heads * (embed_qk_nope + unembed_v_head)
        kv_b_peak = (
            embed_storage_bytes
            + unembed_storage_bytes
            + embed_f32_bytes
            + unembed_f32_bytes
        )

    input_bytes = batch_tokens * hidden_dim * 4
    _stat_expected_file(input_path, input_bytes, "fused attention input")
    command = (
        str(runner),
        "--resident-layout",
        str(layout_path),
        "--layer",
        str(layer),
        "--run-attn-projections",
        "--input-f32",
        str(input_path),
        "--batch-tokens",
        str(batch_tokens),
        "--output-dir",
        str(out_dir),
        "--rms-norm-eps",
        f"{rms_norm_eps:.9g}",
        "--max-resident-matrix-mib",
        str(max_resident_matrix_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    )
    metadata_command = command
    started = time.perf_counter()
    if attention_projection_server_session is not None:
        attention_projection_server_session.submit_batch(
            resident_layout_path=layout_path,
            layer=layer,
            input_f32_path=input_path,
            output_dir=out_dir,
            batch_tokens=batch_tokens,
            rms_norm_eps=rms_norm_eps,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
        )
        metadata_command = (
            str(runner),
            "--run-attn-projections-server-jsonl",
        )
    else:
        _run_command(list(command), echo_output=echo_runner_output)
    elapsed_seconds = time.perf_counter() - started

    _stat_expected_file(input_norm_output, batch_tokens * hidden_dim * 4, "fused input norm")
    _stat_expected_file(q_a_output, batch_tokens * q_a_dim * 4, "fused q_a")
    _stat_expected_file(q_a_norm_output, batch_tokens * q_a_dim * 4, "fused q_a norm")
    _stat_expected_file(q_b_output, batch_tokens * q_b_dim * 4, "fused q_b")
    _stat_expected_file(kv_a_output, batch_tokens * kv_a_dim * 4, "fused kv_a")
    _stat_expected_file(kv_a_norm_output, batch_tokens * kv_lora_dim * 4, "fused kv_a norm")
    if kv_b_tensor is not None:
        _stat_expected_file(kv_b_output, kv_b_output_bytes, "fused kv_b")
    alias_pairs = [
        (input_norm_output, out_dir / "input_layernorm.f32"),
        (q_a_output, out_dir / "q_a_proj.f32"),
        (q_a_norm_output, out_dir / "q_a_layernorm.f32"),
        (q_b_output, out_dir / "q_b_proj.f32"),
        (kv_a_output, out_dir / "kv_a_proj_with_mqa.f32"),
        (kv_a_norm_output, out_dir / "kv_a_layernorm.f32"),
    ]
    if kv_b_tensor is not None:
        alias_pairs.append((kv_b_output, out_dir / "kv_b_proj.f32"))
    for source, alias in alias_pairs:
        shutil.copyfile(source, alias)

    kv_a_lora_bytes, kv_a_rope_bytes, split_peak = _split_f32_row_prefix(
        input_path=kv_a_output,
        prefix_output_path=kv_a_lora_path,
        suffix_output_path=kv_a_rope_path,
        batch_tokens=batch_tokens,
        row_dim=kv_a_dim,
        prefix_dim=kv_lora_dim,
    )
    prefix_command = metadata_command
    input_norm = _resident_rmsnorm_result_metadata(
        runner=runner,
        layout_path=layout_path,
        vector=input_norm_tensor,
        norm_suffix=norm_suffix,
        input_path=input_path,
        output_path=input_norm_output,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        rms_norm_eps=rms_norm_eps,
        command=prefix_command,
    )
    q_a_result = _resident_linear_result_metadata(
        runner=runner,
        layout_path=layout_path,
        matrix=q_a_tensor,
        tensor_suffix=q_a_suffix,
        input_path=input_norm_output,
        output_path=q_a_output,
        layer=layer,
        batch_tokens=batch_tokens,
        in_dim=hidden_dim,
        out_dim=q_a_dim,
        matrix_bytes=q_a_bytes,
        dtype=q_a_dtype,
        backend="fused-metal",
        command=metadata_command,
    )
    kv_a_result = _resident_linear_result_metadata(
        runner=runner,
        layout_path=layout_path,
        matrix=kv_a_tensor,
        tensor_suffix=kv_a_suffix,
        input_path=input_norm_output,
        output_path=kv_a_output,
        layer=layer,
        batch_tokens=batch_tokens,
        in_dim=hidden_dim,
        out_dim=kv_a_dim,
        matrix_bytes=kv_a_bytes,
        dtype=kv_a_dtype,
        backend="fused-metal",
        command=metadata_command,
    )
    prefix = PrefillAttentionPrefixResult(
        runner_path=runner,
        resident_layout_path=layout_path,
        input_path=input_path,
        output_dir=out_dir,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        q_a_dim=q_a_dim,
        kv_a_dim=kv_a_dim,
        input_bytes=input_bytes,
        norm_output_bytes=batch_tokens * hidden_dim * 4,
        q_a_output_bytes=batch_tokens * q_a_dim * 4,
        kv_a_output_bytes=batch_tokens * kv_a_dim * 4,
        estimated_peak_bytes=max(
            input_norm.estimated_peak_bytes,
            q_a_result.estimated_peak_bytes,
            kv_a_result.estimated_peak_bytes,
        ),
        input_layernorm=input_norm,
        q_a_proj=q_a_result,
        kv_a_proj_with_mqa=kv_a_result,
    )
    q_a_norm = _resident_rmsnorm_result_metadata(
        runner=runner,
        layout_path=layout_path,
        vector=q_a_norm_tensor,
        norm_suffix=q_a_norm_suffix,
        input_path=q_a_output,
        output_path=q_a_norm_output,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=q_a_dim,
        rms_norm_eps=rms_norm_eps,
        command=metadata_command,
    )
    q_b_result = _resident_linear_result_metadata(
        runner=runner,
        layout_path=layout_path,
        matrix=q_b_tensor,
        tensor_suffix=q_b_suffix,
        input_path=q_a_norm_output,
        output_path=q_b_output,
        layer=layer,
        batch_tokens=batch_tokens,
        in_dim=q_a_dim,
        out_dim=q_b_dim,
        matrix_bytes=q_b_bytes,
        dtype=q_b_dtype,
        backend="fused-metal",
        command=metadata_command,
    )
    kv_a_norm = _resident_rmsnorm_result_metadata(
        runner=runner,
        layout_path=layout_path,
        vector=kv_a_norm_tensor,
        norm_suffix=kv_a_norm_suffix,
        input_path=kv_a_lora_path,
        output_path=kv_a_norm_output,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=kv_lora_dim,
        rms_norm_eps=rms_norm_eps,
        command=metadata_command,
    )
    if kv_b_tensor is not None:
        assert kv_b_tensor is not None
        kv_b_result = _resident_linear_result_metadata(
            runner=runner,
            layout_path=layout_path,
            matrix=kv_b_tensor,
            tensor_suffix=kv_b_suffix,
            input_path=kv_a_norm_output,
            output_path=kv_b_output,
            layer=layer,
            batch_tokens=batch_tokens,
            in_dim=kv_lora_dim,
            out_dim=kv_b_dim,
            matrix_bytes=kv_b_bytes,
            dtype=kv_b_dtype,
            backend="fused-metal",
            command=metadata_command,
        )
        kv_b_peak = kv_b_result.estimated_peak_bytes

    estimated_peak = max(
        prefix.estimated_peak_bytes,
        q_a_norm.estimated_peak_bytes,
        q_b_result.estimated_peak_bytes,
        split_peak,
        kv_a_norm.estimated_peak_bytes,
        kv_b_peak,
    )
    return PrefillAttentionProjectionBatchResult(
        runner_path=runner,
        resident_layout_path=layout_path,
        input_path=input_path,
        output_dir=out_dir,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=hidden_dim,
        q_a_dim=q_a_dim,
        q_b_dim=q_b_dim,
        kv_a_dim=kv_a_dim,
        kv_lora_dim=kv_lora_dim,
        kv_rope_dim=kv_rope_dim,
        kv_b_dim=kv_b_dim,
        attention_value_source=attention_value_source,
        input_bytes=input_bytes,
        q_b_output_bytes=batch_tokens * q_b_dim * 4,
        kv_a_lora_bytes=kv_a_lora_bytes,
        kv_a_rope_bytes=kv_a_rope_bytes,
        kv_b_output_bytes=kv_b_output_bytes,
        split_peak_bytes=split_peak,
        estimated_peak_bytes=estimated_peak,
        prefix=prefix,
        q_a_layernorm=q_a_norm,
        q_b_proj=q_b_result,
        kv_a_lora_path=kv_a_lora_path,
        kv_a_rope_path=kv_a_rope_path,
        kv_a_layernorm=kv_a_norm,
        kv_b_proj=kv_b_result,
    )


def run_prefill_attention_projection_batch(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_dir: str | Path,
    batch_tokens: int,
    norm_suffix: str = ".input_layernorm.weight",
    q_a_suffix: str = ".self_attn.q_a_proj.weight",
    q_a_norm_suffix: str = ".self_attn.q_a_layernorm.weight",
    q_b_suffix: str = ".self_attn.q_b_proj.weight",
    kv_a_suffix: str = ".self_attn.kv_a_proj_with_mqa.weight",
    kv_a_norm_suffix: str = ".self_attn.kv_a_layernorm.weight",
    kv_b_suffix: str = ".self_attn.kv_b_proj.weight",
    rms_norm_eps: float = 1e-5,
    max_resident_matrix_mib: int = 512,
    max_runner_scratch_mib: int = 4096,
    prefill_linear_backend: str = "custom-metal",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    echo_runner_output: bool = True,
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None,
    attention_projection_server_session: AttentionProjectionsServerSession | None = None,
    rmsnorm_server_session: ResidentBatchRMSNormServerSession | None = None,
) -> PrefillAttentionProjectionBatchResult:
    """Run bounded GLM batch attention projections through q_b and kv_b."""

    if (
        type(batch_tokens) is int
        and batch_tokens > 0
        and (
            batch_tokens == 1
            or _batch_fused_attention_projections_enabled()
        )
        and Path(runner_path).name == "largerlm-runner"
        and prefill_linear_backend in {"custom-metal", "auto"}
        and norm_suffix == ".input_layernorm.weight"
        and q_a_suffix == ".self_attn.q_a_proj.weight"
        and q_a_norm_suffix == ".self_attn.q_a_layernorm.weight"
        and q_b_suffix == ".self_attn.q_b_proj.weight"
        and kv_a_suffix == ".self_attn.kv_a_proj_with_mqa.weight"
        and kv_a_norm_suffix == ".self_attn.kv_a_layernorm.weight"
        and kv_b_suffix == ".self_attn.kv_b_proj.weight"
    ):
        return _run_prefill_attention_projection_batch_fused_runner(
            runner_path=runner_path,
            resident_layout_path=resident_layout_path,
            layer=layer,
            input_f32_path=input_f32_path,
            output_dir=output_dir,
            batch_tokens=batch_tokens,
            norm_suffix=norm_suffix,
            q_a_suffix=q_a_suffix,
            q_a_norm_suffix=q_a_norm_suffix,
            q_b_suffix=q_b_suffix,
            kv_a_suffix=kv_a_suffix,
            kv_a_norm_suffix=kv_a_norm_suffix,
            kv_b_suffix=kv_b_suffix,
            rms_norm_eps=rms_norm_eps,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            echo_runner_output=echo_runner_output,
            attention_projection_server_session=attention_projection_server_session,
        )

    out_dir = Path(output_dir)
    prefix = run_prefill_attention_prefix_batch(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        input_f32_path=input_f32_path,
        output_dir=out_dir,
        batch_tokens=batch_tokens,
        norm_suffix=norm_suffix,
        q_a_suffix=q_a_suffix,
        kv_a_suffix=kv_a_suffix,
        rms_norm_eps=rms_norm_eps,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
        rmsnorm_server_session=rmsnorm_server_session,
    )

    q_a_norm_output = out_dir / "q_a_layernorm.f32"
    q_b_output = out_dir / "q_b_proj.f32"
    q_a_norm = run_resident_batch_rmsnorm(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        norm_suffix=q_a_norm_suffix,
        input_f32_path=prefix.q_a_proj.output_path,
        output_f32_path=q_a_norm_output,
        batch_tokens=batch_tokens,
        rms_norm_eps=rms_norm_eps,
        max_runner_scratch_mib=max_runner_scratch_mib,
        echo_runner_output=echo_runner_output,
        rmsnorm_server_session=rmsnorm_server_session,
    )
    q_b = run_resident_batch_linear(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        tensor_suffix=q_b_suffix,
        input_f32_path=q_a_norm_output,
        output_f32_path=q_b_output,
        batch_tokens=batch_tokens,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
    )

    layout = _load_json(prefix.resident_layout_path)
    kv_a_norm_tensor = _find_layer_vector(layout, layer=layer, suffix=kv_a_norm_suffix)
    kv_lora_dim = _tensor_shape1(
        kv_a_norm_tensor,
        str(kv_a_norm_tensor.get("name") or "kv_a_layernorm"),
    )
    kv_rope_dim = prefix.kv_a_dim - kv_lora_dim
    if kv_rope_dim < 0:
        raise PrefillExecuteError(
            f"kv_a output dim {prefix.kv_a_dim} is smaller than kv_lora_dim "
            f"{kv_lora_dim}"
        )

    kv_a_lora_path = out_dir / "kv_a_lora.f32"
    kv_a_rope_path = out_dir / "kv_a_rope.f32"
    kv_a_lora_bytes, kv_a_rope_bytes, split_peak = _split_f32_row_prefix(
        input_path=prefix.kv_a_proj_with_mqa.output_path,
        prefix_output_path=kv_a_lora_path,
        suffix_output_path=kv_a_rope_path,
        batch_tokens=batch_tokens,
        row_dim=prefix.kv_a_dim,
        prefix_dim=kv_lora_dim,
    )

    kv_a_norm_output = out_dir / "kv_a_layernorm.f32"
    kv_b_output = out_dir / "kv_b_proj.f32"
    kv_a_norm = run_resident_batch_rmsnorm(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        norm_suffix=kv_a_norm_suffix,
        input_f32_path=kv_a_lora_path,
        output_f32_path=kv_a_norm_output,
        batch_tokens=batch_tokens,
        rms_norm_eps=rms_norm_eps,
        max_runner_scratch_mib=max_runner_scratch_mib,
        echo_runner_output=echo_runner_output,
        rmsnorm_server_session=rmsnorm_server_session,
    )
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise PrefillExecuteError("resident layout missing weight_file")
    weight_path = prefix.resident_layout_path.parent / weight_file
    kv_b_tensor = _find_layer_tensor_optional(layout, layer=layer, suffix=kv_b_suffix)
    attention_value_source = "kv_b_proj"
    kv_b: ResidentBatchLinearResult | None = None
    if kv_b_tensor is not None:
        kv_b = run_resident_batch_linear(
            runner_path=runner_path,
            resident_layout_path=resident_layout_path,
            layer=layer,
            tensor_suffix=kv_b_suffix,
            input_f32_path=kv_a_norm_output,
            output_f32_path=kv_b_output,
            batch_tokens=batch_tokens,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            prefill_linear_backend=prefill_linear_backend,
            prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
            prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
            echo_runner_output=echo_runner_output,
            resident_linear_server_session=resident_linear_server_session,
        )
        kv_b_dim = kv_b.out_dim
        kv_b_output_bytes = kv_b.output_bytes
        kv_b_peak = kv_b.estimated_peak_bytes
    else:
        embed_q = _find_layer_matrix(
            layout,
            layer=layer,
            suffix=".self_attn.embed_q.weight",
        )
        unembed_out = _find_layer_matrix(
            layout,
            layer=layer,
            suffix=".self_attn.unembed_out.weight",
        )
        (
            embed_heads,
            embed_kv_lora,
            embed_qk_nope,
            embed_storage_bytes,
            embed_f32_bytes,
        ) = _tensor_shape3_value_source(
            layout,
            embed_q,
            "embed_q.weight",
            weight_path,
        )
        (
            unembed_heads,
            unembed_v_head,
            unembed_kv_lora,
            unembed_storage_bytes,
            unembed_f32_bytes,
        ) = _tensor_shape3_value_source(
            layout,
            unembed_out,
            "unembed_out.weight",
            weight_path,
        )
        if (
            embed_kv_lora != kv_lora_dim
            or unembed_kv_lora != kv_lora_dim
            or embed_heads != unembed_heads
        ):
            raise PrefillExecuteError(
                "absorbed attention alias dims are inconsistent: "
                f"embed_q=[{embed_heads},{embed_kv_lora},{embed_qk_nope}], "
                f"unembed_out=[{unembed_heads},{unembed_v_head},{unembed_kv_lora}], "
                f"kv_lora_dim={kv_lora_dim}"
            )
        max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
        max_alias_bytes = max(
            embed_storage_bytes,
            unembed_storage_bytes,
            embed_f32_bytes,
            unembed_f32_bytes,
        )
        if max_alias_bytes > max_matrix_bytes:
            raise PrefillExecuteError(
                f"absorbed attention alias has {max_alias_bytes} bytes, exceeds limit "
                f"{max_matrix_bytes}"
            )
        attention_value_source = "absorbed-alias"
        kv_b_dim = embed_heads * (embed_qk_nope + unembed_v_head)
        kv_b_output_bytes = 0
        kv_b_peak = embed_storage_bytes + unembed_storage_bytes + embed_f32_bytes + unembed_f32_bytes

    estimated_peak = max(
        prefix.estimated_peak_bytes,
        q_a_norm.estimated_peak_bytes,
        q_b.estimated_peak_bytes,
        split_peak,
        kv_a_norm.estimated_peak_bytes,
        kv_b_peak,
    )
    return PrefillAttentionProjectionBatchResult(
        runner_path=prefix.runner_path,
        resident_layout_path=prefix.resident_layout_path,
        input_path=prefix.input_path,
        output_dir=out_dir,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=prefix.hidden_dim,
        q_a_dim=prefix.q_a_dim,
        q_b_dim=q_b.out_dim,
        kv_a_dim=prefix.kv_a_dim,
        kv_lora_dim=kv_lora_dim,
        kv_rope_dim=kv_rope_dim,
        kv_b_dim=kv_b_dim,
        attention_value_source=attention_value_source,
        input_bytes=prefix.input_bytes,
        q_b_output_bytes=q_b.output_bytes,
        kv_a_lora_bytes=kv_a_lora_bytes,
        kv_a_rope_bytes=kv_a_rope_bytes,
        kv_b_output_bytes=kv_b_output_bytes,
        split_peak_bytes=split_peak,
        estimated_peak_bytes=estimated_peak,
        prefix=prefix,
        q_a_layernorm=q_a_norm,
        q_b_proj=q_b,
        kv_a_lora_path=kv_a_lora_path,
        kv_a_rope_path=kv_a_rope_path,
        kv_a_layernorm=kv_a_norm,
        kv_b_proj=kv_b,
    )


def write_prefill_kv_cache_batch(
    *,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    start_position: int,
    batch_tokens: int,
    max_cache_file_mib: float = 32768.0,
    max_cache_write_mib: float = 4096.0,
) -> PrefillCacheWriteResult:
    if layer < 0:
        raise PrefillExecuteError("layer must be non-negative")
    if start_position < 0:
        raise PrefillExecuteError("start_position must be non-negative")
    if batch_tokens <= 0:
        raise PrefillExecuteError("batch_tokens must be positive")
    if max_cache_file_mib <= 0 or max_cache_write_mib <= 0:
        raise PrefillExecuteError("cache file and write limits must be positive")

    layout_path = Path(cache_layout_path)
    cache_path = Path(cache_file_path)
    input_path = Path(input_f32_path)
    layout = load_decode_cache_layout(layout_path)
    max_cache_file_bytes = int(max_cache_file_mib * 1024 * 1024)
    max_cache_write_bytes = int(max_cache_write_mib * 1024 * 1024)
    if layout.total_bytes > max_cache_file_bytes:
        raise PrefillExecuteError(
            f"cache layout total {layout.total_bytes} bytes exceeds limit "
            f"{max_cache_file_bytes}"
        )
    try:
        cache_file_bytes = cache_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat cache file {cache_path}: {exc}") from exc
    if cache_file_bytes < layout.total_bytes:
        raise PrefillExecuteError(
            f"cache file has {cache_file_bytes} bytes, expected at least "
            f"{layout.total_bytes}"
        )
    if cache_file_bytes > max_cache_file_bytes:
        raise PrefillExecuteError(
            f"cache file has {cache_file_bytes} bytes, exceeds limit "
            f"{max_cache_file_bytes}"
        )

    segment = next(
        (
            item
            for item in layout.segments
            if item.kind == "mla_kv" and item.layer == layer
        ),
        None,
    )
    if segment is None:
        raise PrefillExecuteError(f"mla_kv cache segment for layer {layer} not found")
    if start_position + batch_tokens > segment.max_context_tokens:
        raise PrefillExecuteError(
            f"cache write positions [{start_position}, {start_position + batch_tokens}) "
            f"exceed max context {segment.max_context_tokens}"
        )
    token_stride_bytes = segment.token_stride_bytes
    encoded_bytes = batch_tokens * token_stride_bytes
    if encoded_bytes > max_cache_write_bytes:
        raise PrefillExecuteError(
            f"cache write {encoded_bytes} bytes exceeds limit {max_cache_write_bytes}"
        )

    input_row_bytes = segment.width * 4
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat cache input {input_path}: {exc}") from exc
    expected_input_bytes = batch_tokens * input_row_bytes
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"input bytes {input_bytes} do not match batch_tokens*cache_width*f32 "
            f"({expected_input_bytes})"
        )

    first_write_offset = segment.offset + start_position * token_stride_bytes
    final_write_end = first_write_offset + encoded_bytes
    if final_write_end > segment.offset + segment.total_bytes:
        raise PrefillExecuteError("cache write exceeds segment bounds")
    if final_write_end > layout.total_bytes:
        raise PrefillExecuteError("cache write exceeds layout total bytes")

    transient_bytes_per_token = input_row_bytes + token_stride_bytes
    configured_chunk_bytes = _prefill_cache_write_chunk_bytes()
    write_chunk_bytes = max(
        transient_bytes_per_token,
        min(configured_chunk_bytes, max_cache_write_bytes),
    )
    tokens_per_chunk = max(1, write_chunk_bytes // transient_bytes_per_token)
    write_chunk_tokens = min(batch_tokens, tokens_per_chunk)
    estimated_peak_bytes = write_chunk_tokens * transient_bytes_per_token
    write_chunks = 0
    try:
        with input_path.open("rb") as source, cache_path.open("r+b") as cache:
            token_index = 0
            while token_index < batch_tokens:
                chunk_tokens = min(tokens_per_chunk, batch_tokens - token_index)
                chunk_input_bytes = chunk_tokens * input_row_bytes
                rows = source.read(chunk_input_bytes)
                if len(rows) != chunk_input_bytes:
                    raise PrefillExecuteError(
                        f"failed to read full cache input chunk from {input_path}"
                    )
                encoded = _encode_cache_rows(
                    rows,
                    width=segment.width,
                    batch_tokens=chunk_tokens,
                    dtype=segment.dtype,
                    dtype_bytes=segment.dtype_bytes,
                )
                expected_encoded = chunk_tokens * token_stride_bytes
                if len(encoded) != expected_encoded:
                    raise PrefillExecuteError(
                        f"encoded cache chunk has {len(encoded)} bytes, expected "
                        f"{expected_encoded}"
                    )
                write_offset = (
                    segment.offset
                    + (start_position + token_index) * token_stride_bytes
                )
                cache.seek(write_offset)
                cache.write(encoded)
                token_index += chunk_tokens
                write_chunks += 1
    except OSError as exc:
        raise PrefillExecuteError(f"failed to write prefill cache rows: {exc}") from exc

    encoder = "copy_f32" if segment.dtype in {"F32", "float32"} else "bitcast_bf16"
    return PrefillCacheWriteResult(
        cache_layout_path=layout_path,
        cache_file_path=cache_path,
        input_path=input_path,
        layer=layer,
        start_position=start_position,
        batch_tokens=batch_tokens,
        width=segment.width,
        dtype=segment.dtype,
        dtype_bytes=segment.dtype_bytes,
        token_stride_bytes=token_stride_bytes,
        segment_offset=segment.offset,
        first_write_offset=first_write_offset,
        input_bytes=input_bytes,
        encoded_bytes=encoded_bytes,
        max_cache_file_bytes=max_cache_file_bytes,
        max_cache_write_bytes=max_cache_write_bytes,
        write_chunk_tokens=write_chunk_tokens,
        write_chunk_bytes=write_chunk_bytes,
        write_chunks=write_chunks,
        estimated_peak_bytes=estimated_peak_bytes,
        encoder=encoder,
    )


def run_prefill_rope_batch(
    *,
    runner_path: str | Path,
    q_b_f32_path: str | Path,
    k_rope_f32_path: str | Path,
    output_dir: str | Path,
    batch_tokens: int,
    num_heads: int,
    qk_nope_dim: int,
    rope_dim: int,
    start_position: int,
    rope_theta: float = 10000.0,
    rope_interleave: bool = False,
    max_runner_scratch_mib: int = 4096,
    echo_runner_output: bool = True,
    rope_split_server_session: RopeSplitBatchServerSession | None = None,
) -> PrefillRopeBatchResult:
    if batch_tokens <= 0 or num_heads <= 0 or qk_nope_dim <= 0 or rope_dim <= 0:
        raise PrefillExecuteError("batch, head, qk_nope, and rope dims must be positive")
    if rope_dim % 2 != 0:
        raise PrefillExecuteError("rope_dim must be even")
    if start_position < 0:
        raise PrefillExecuteError("start_position must be non-negative")
    if rope_theta <= 0:
        raise PrefillExecuteError("rope_theta must be positive")
    if max_runner_scratch_mib <= 0:
        raise PrefillExecuteError("scratch limit must be positive")

    runner = Path(runner_path)
    if not runner.exists():
        raise PrefillExecuteError(f"runner not found: {runner}")
    q_b_path = Path(q_b_f32_path)
    k_rope_path = Path(k_rope_f32_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q_nope_path = out_dir / "q_nope.f32"
    q_rope_path = out_dir / "q_rope.f32"
    q_rope_rotated_path = out_dir / "q_rope_rotated.f32"
    k_rope_rotated_path = out_dir / "k_rope_rotated.f32"

    q_head_dim = qk_nope_dim + rope_dim
    q_b_expected_bytes = batch_tokens * num_heads * q_head_dim * 4
    q_nope_bytes = batch_tokens * num_heads * qk_nope_dim * 4
    q_rope_bytes = batch_tokens * num_heads * rope_dim * 4
    k_rope_bytes = batch_tokens * rope_dim * 4
    try:
        q_b_input_bytes = q_b_path.stat().st_size
        k_rope_input_bytes = k_rope_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat RoPE batch inputs: {exc}") from exc
    if q_b_input_bytes != q_b_expected_bytes:
        raise PrefillExecuteError(
            f"q_b input bytes {q_b_input_bytes} do not match expected "
            f"{q_b_expected_bytes}"
        )
    if k_rope_input_bytes != k_rope_bytes:
        raise PrefillExecuteError(
            f"k_rope input bytes {k_rope_input_bytes} do not match expected "
            f"{k_rope_bytes}"
        )
    use_fused_split = (
        batch_tokens > 1
        and runner.name == "largerlm-runner"
        and _fused_rope_split_batch_enabled()
    )
    split_peak = q_b_input_bytes + q_nope_bytes + q_rope_bytes
    runner_peak = (
        q_b_input_bytes + k_rope_input_bytes + q_nope_bytes + 2 * q_rope_bytes + k_rope_bytes
        if use_fused_split
        else 2 * q_rope_bytes + 2 * k_rope_bytes
    )
    estimated_peak = max(split_peak, runner_peak)
    max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    if estimated_peak > max_scratch_bytes:
        raise PrefillExecuteError(
            f"estimated peak {estimated_peak} bytes exceeds scratch limit "
            f"{max_scratch_bytes}"
        )

    cmd: list[str]
    if batch_tokens == 1 and runner.name == "largerlm-runner":
        _split_q_b_batch(
            input_path=q_b_path,
            q_nope_path=q_nope_path,
            q_rope_path=q_rope_path,
            batch_tokens=batch_tokens,
            num_heads=num_heads,
            qk_nope_dim=qk_nope_dim,
            rope_dim=rope_dim,
        )
        cmd = [
            "python-rope-singleton",
            "--q-f32",
            str(q_rope_path),
            "--k-f32",
            str(k_rope_path),
            "--output-q-f32",
            str(q_rope_rotated_path),
            "--output-k-f32",
            str(k_rope_rotated_path),
            "--num-heads",
            str(num_heads),
            "--rope-dim",
            str(rope_dim),
            "--position",
            str(start_position),
            "--rope-theta",
            f"{rope_theta:.9g}",
        ]
        if rope_interleave:
            cmd.append("--rope-interleave")
        _write_singleton_rope_python(
            q_rope_path=q_rope_path,
            k_rope_path=k_rope_path,
            q_rope_rotated_path=q_rope_rotated_path,
            k_rope_rotated_path=k_rope_rotated_path,
            num_heads=num_heads,
            rope_dim=rope_dim,
            position=start_position,
            theta=rope_theta,
            interleave=rope_interleave,
        )
    elif use_fused_split:
        cmd = [
            str(runner),
            "--run-rope-split-batch",
            "--q-b-f32",
            str(q_b_path),
            "--k-f32",
            str(k_rope_path),
            "--output-q-nope-f32",
            str(q_nope_path),
            "--output-q-rope-f32",
            str(q_rope_path),
            "--output-q-f32",
            str(q_rope_rotated_path),
            "--output-k-f32",
            str(k_rope_rotated_path),
            "--num-heads",
            str(num_heads),
            "--qk-nope-dim",
            str(qk_nope_dim),
            "--rope-dim",
            str(rope_dim),
            "--start-position",
            str(start_position),
            "--batch-tokens",
            str(batch_tokens),
            "--rope-theta",
            f"{rope_theta:.9g}",
            "--max-runner-scratch-mib",
            str(max_runner_scratch_mib),
        ]
        if rope_interleave:
            cmd.append("--rope-interleave")
        if rope_split_server_session is not None:
            rope_split_server_session.submit_batch(
                q_b_f32_path=q_b_path,
                k_f32_path=k_rope_path,
                output_q_nope_f32_path=q_nope_path,
                output_q_rope_f32_path=q_rope_path,
                output_q_f32_path=q_rope_rotated_path,
                output_k_f32_path=k_rope_rotated_path,
                num_heads=num_heads,
                qk_nope_dim=qk_nope_dim,
                rope_dim=rope_dim,
                start_position=start_position,
                batch_tokens=batch_tokens,
                rope_theta=rope_theta,
                rope_interleave=rope_interleave,
                max_runner_scratch_mib=max_runner_scratch_mib,
            )
            cmd = [str(runner), "--run-rope-split-batch-server-jsonl"]
        else:
            _run_command(cmd, echo_output=echo_runner_output)
    else:
        _split_q_b_batch(
            input_path=q_b_path,
            q_nope_path=q_nope_path,
            q_rope_path=q_rope_path,
            batch_tokens=batch_tokens,
            num_heads=num_heads,
            qk_nope_dim=qk_nope_dim,
            rope_dim=rope_dim,
        )
        cmd = [
            str(runner),
            "--run-rope-batch",
            "--q-f32",
            str(q_rope_path),
            "--k-f32",
            str(k_rope_path),
            "--output-q-f32",
            str(q_rope_rotated_path),
            "--output-k-f32",
            str(k_rope_rotated_path),
            "--num-heads",
            str(num_heads),
            "--rope-dim",
            str(rope_dim),
            "--start-position",
            str(start_position),
            "--batch-tokens",
            str(batch_tokens),
            "--rope-theta",
            f"{rope_theta:.9g}",
            "--max-runner-scratch-mib",
            str(max_runner_scratch_mib),
        ]
        if rope_interleave:
            cmd.append("--rope-interleave")
        _run_command(cmd, echo_output=echo_runner_output)
    for path, expected in (
        (q_nope_path, q_nope_bytes),
        (q_rope_path, q_rope_bytes),
        (q_rope_rotated_path, q_rope_bytes),
        (k_rope_rotated_path, k_rope_bytes),
    ):
        try:
            actual = path.stat().st_size
        except OSError as exc:
            raise PrefillExecuteError(f"failed to stat RoPE output {path}: {exc}") from exc
        if actual != expected:
            raise PrefillExecuteError(
                f"RoPE output {path} has {actual} bytes, expected {expected}"
            )

    return PrefillRopeBatchResult(
        runner_path=runner,
        q_b_input_path=q_b_path,
        k_rope_input_path=k_rope_path,
        output_dir=out_dir,
        q_nope_path=q_nope_path,
        q_rope_path=q_rope_path,
        q_rope_rotated_path=q_rope_rotated_path,
        k_rope_rotated_path=k_rope_rotated_path,
        batch_tokens=batch_tokens,
        num_heads=num_heads,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
        start_position=start_position,
        rope_theta=rope_theta,
        rope_interleave=rope_interleave,
        q_b_input_bytes=q_b_input_bytes,
        k_rope_input_bytes=k_rope_input_bytes,
        q_nope_bytes=q_nope_bytes,
        q_rope_bytes=q_rope_bytes,
        k_rope_bytes=k_rope_bytes,
        split_peak_bytes=split_peak,
        estimated_peak_bytes=estimated_peak,
        command=tuple(cmd),
    )


def run_prefill_mla_attention_batch(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    layer: int,
    q_nope_f32_path: str | Path,
    q_rope_f32_path: str | Path,
    output_f32_path: str | Path,
    context_length: int,
    start_position: int,
    batch_tokens: int,
    num_heads: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    mla_kv_b_cache_dir: str | Path | None = None,
    mla_key_cache: bool = False,
    mla_value_cache: bool = True,
    indices_u32_path: str | Path | None = None,
    index_topk: int | None = None,
    kv_lora_dim: int | None = None,
    cache_position_offset: int = 0,
    attention_scale: float | None = None,
    rope_theta: float = 10000.0,
    rope_interleave: bool = False,
    max_cache_file_mib: int = 32768,
    max_cache_read_mib: int = 256,
    max_resident_matrix_mib: int = 512,
    max_runner_scratch_mib: int = 4096,
    echo_runner_output: bool = True,
    mla_attention_server_session: MLAAttentionBatchServerSession | None = None,
) -> PrefillMLAAttentionBatchResult:
    layer = _require_int(layer, "layer")
    context_length = _require_int(context_length, "context_length")
    start_position = _require_int(start_position, "start_position")
    batch_tokens = _require_int(batch_tokens, "batch_tokens")
    num_heads = _require_int(num_heads, "num_heads")
    qk_nope_dim = _require_int(qk_nope_dim, "qk_nope_dim")
    rope_dim = _require_int(rope_dim, "rope_dim")
    v_head_dim = _require_int(v_head_dim, "v_head_dim")
    cache_position_offset = _require_int(
        cache_position_offset, "cache_position_offset"
    )
    if kv_lora_dim is not None:
        kv_lora_dim = _require_int(kv_lora_dim, "kv_lora_dim")
    if index_topk is not None:
        index_topk = _require_int(index_topk, "index_topk")
    if attention_scale is not None:
        attention_scale = _require_number(attention_scale, "attention_scale")
    rope_theta = _require_number(rope_theta, "rope_theta")
    max_cache_file_mib = _require_number(max_cache_file_mib, "max_cache_file_mib")
    max_cache_read_mib = _require_number(max_cache_read_mib, "max_cache_read_mib")
    max_resident_matrix_mib = _require_number(
        max_resident_matrix_mib, "max_resident_matrix_mib"
    )
    max_runner_scratch_mib = _require_number(
        max_runner_scratch_mib, "max_runner_scratch_mib"
    )
    if layer < 0:
        raise PrefillExecuteError("layer must be non-negative")
    if context_length <= 0 or start_position < 0 or batch_tokens <= 0:
        raise PrefillExecuteError("context, start position, and batch size are invalid")
    if start_position + batch_tokens > context_length:
        raise PrefillExecuteError("start_position + batch_tokens exceeds context_length")
    if num_heads <= 0 or qk_nope_dim <= 0 or rope_dim <= 0 or v_head_dim <= 0:
        raise PrefillExecuteError("attention dimensions must be positive")
    if rope_dim % 2 != 0:
        raise PrefillExecuteError("rope_dim must be even")
    if kv_lora_dim is not None and kv_lora_dim <= 0:
        raise PrefillExecuteError("kv_lora_dim must be positive when provided")
    indexed = indices_u32_path is not None or index_topk is not None
    if (indices_u32_path is None) != (index_topk is None):
        raise PrefillExecuteError("indices_u32_path and index_topk must be provided together")
    if index_topk is not None and index_topk <= 0:
        raise PrefillExecuteError("index_topk must be positive when provided")
    if isinstance(mla_kv_b_cache_dir, bool):
        raise PrefillExecuteError("mla_kv_b_cache_dir must be a path when provided")
    if type(mla_key_cache) is not bool:
        raise PrefillExecuteError("mla_key_cache must be a boolean")
    if type(mla_value_cache) is not bool:
        raise PrefillExecuteError("mla_value_cache must be a boolean")
    if _mla_value_cache_disabled_by_env():
        mla_value_cache = False
    mla_kv_b_cache_path = (
        Path(mla_kv_b_cache_dir) if mla_kv_b_cache_dir is not None else None
    )
    if mla_kv_b_cache_path is not None:
        try:
            mla_kv_b_cache_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to create MLA kv_b cache dir {mla_kv_b_cache_path}: {exc}"
            ) from exc
    if cache_position_offset < 0:
        raise PrefillExecuteError("cache_position_offset must be non-negative")
    if attention_scale is not None and attention_scale <= 0:
        raise PrefillExecuteError("attention_scale must be positive when provided")
    if rope_theta <= 0:
        raise PrefillExecuteError("rope_theta must be positive")
    if (
        max_cache_file_mib <= 0
        or max_cache_read_mib <= 0
        or max_resident_matrix_mib <= 0
        or max_runner_scratch_mib <= 0
    ):
        raise PrefillExecuteError("memory limits must be positive")

    runner = Path(runner_path)
    if not runner.exists():
        raise PrefillExecuteError(f"runner not found: {runner}")
    resident_layout = Path(resident_layout_path)
    resident = _load_json(resident_layout)
    weight_file = resident.get("weight_file")
    if not isinstance(weight_file, str):
        raise PrefillExecuteError("resident layout missing weight_file")
    weight_path = resident_layout.parent / weight_file
    if not weight_path.exists():
        raise PrefillExecuteError(f"resident weight file not found: {weight_path}")
    max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
    kv_b = _find_layer_tensor_optional(
        resident,
        layer=layer,
        suffix=".self_attn.kv_b_proj.weight",
    )
    expected_kv_b_out = num_heads * (qk_nope_dim + v_head_dim)
    kv_b_source_f32_bytes = 0
    if kv_b is not None:
        attention_value_source = "kv_b_proj"
        kv_b_out, kv_b_in = _tensor_shape2(kv_b, str(kv_b.get("name") or "kv_b"))
        _check_tensor_backing_span(
            weight_path,
            kv_b,
            str(kv_b.get("name") or "kv_b"),
        )
        resolved_kv_lora = kv_lora_dim or kv_b_in
        if kv_b_in != resolved_kv_lora or kv_b_out != expected_kv_b_out:
            raise PrefillExecuteError(
                f"kv_b shape [{kv_b_out},{kv_b_in}] does not match expected "
                f"[{expected_kv_b_out},{resolved_kv_lora}]"
            )
        kv_b_matrix_bytes = _int_field(
            kv_b,
            "size",
            str(kv_b.get("name") or "kv_b"),
        )
        if kv_b_matrix_bytes > max_matrix_bytes:
            raise PrefillExecuteError(
                f"kv_b matrix has {kv_b_matrix_bytes} bytes, exceeds limit "
                f"{max_matrix_bytes}"
            )
    else:
        attention_value_source = (
            "absorbed-alias-cache"
            if mla_kv_b_cache_path is not None
            else "absorbed-alias"
        )
        embed_q = _find_layer_matrix(
            resident,
            layer=layer,
            suffix=".self_attn.embed_q.weight",
        )
        unembed_out = _find_layer_matrix(
            resident,
            layer=layer,
            suffix=".self_attn.unembed_out.weight",
        )
        (
            embed_heads,
            embed_kv_lora,
            embed_qk_nope,
            embed_storage_bytes,
            embed_f32_bytes,
        ) = _tensor_shape3_value_source(
            resident,
            embed_q,
            "embed_q.weight",
            weight_path,
        )
        (
            unembed_heads,
            unembed_v_head,
            unembed_kv_lora,
            unembed_storage_bytes,
            unembed_f32_bytes,
        ) = _tensor_shape3_value_source(
            resident,
            unembed_out,
            "unembed_out.weight",
            weight_path,
        )
        resolved_kv_lora = kv_lora_dim or embed_kv_lora
        if (
            embed_heads != num_heads
            or unembed_heads != num_heads
            or embed_kv_lora != resolved_kv_lora
            or unembed_kv_lora != resolved_kv_lora
            or embed_qk_nope != qk_nope_dim
            or unembed_v_head != v_head_dim
        ):
            raise PrefillExecuteError(
                "absorbed attention alias shapes do not match expected "
                f"embed_q=[{num_heads},{resolved_kv_lora},{qk_nope_dim}], "
                f"unembed_out=[{num_heads},{v_head_dim},{resolved_kv_lora}]"
            )
        kv_b_out = expected_kv_b_out
        kv_b_in = resolved_kv_lora
        kv_b_matrix_bytes = embed_storage_bytes + unembed_storage_bytes
        kv_b_source_f32_bytes = embed_f32_bytes + unembed_f32_bytes
        max_alias_matrix_bytes = max(
            embed_storage_bytes,
            unembed_storage_bytes,
            embed_f32_bytes,
            unembed_f32_bytes,
        )
        if max_alias_matrix_bytes > max_matrix_bytes:
            raise PrefillExecuteError(
                f"absorbed attention alias has {max_alias_matrix_bytes} bytes, "
                f"exceeds limit {max_matrix_bytes}"
            )

    cache_layout_p = Path(cache_layout_path)
    cache_file_p = Path(cache_file_path)
    cache_layout = load_decode_cache_layout(cache_layout_p)
    max_cache_file_bytes = max_cache_file_mib * 1024 * 1024
    max_cache_read_bytes = max_cache_read_mib * 1024 * 1024
    if cache_layout.total_bytes > max_cache_file_bytes:
        raise PrefillExecuteError(
            f"cache layout total {cache_layout.total_bytes} bytes exceeds limit "
            f"{max_cache_file_bytes}"
        )
    try:
        cache_file_bytes = cache_file_p.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat cache file {cache_file_p}: {exc}") from exc
    if cache_file_bytes < cache_layout.total_bytes:
        raise PrefillExecuteError(
            f"cache file has {cache_file_bytes} bytes, expected at least "
            f"{cache_layout.total_bytes}"
        )
    segment = next(
        (
            item
            for item in cache_layout.segments
            if item.kind == "mla_kv" and item.layer == layer
        ),
        None,
    )
    if segment is None:
        raise PrefillExecuteError(f"mla_kv cache segment for layer {layer} not found")
    cache_width = resolved_kv_lora + rope_dim
    if segment.width != cache_width:
        raise PrefillExecuteError(
            f"cache width {segment.width} does not match kv_lora+rope {cache_width}"
        )
    if context_length > segment.max_context_tokens:
        raise PrefillExecuteError(
            f"context_length {context_length} exceeds max context "
            f"{segment.max_context_tokens}"
        )
    indices_path = Path(indices_u32_path) if indices_u32_path is not None else None
    indices_u32_bytes = 0
    if indexed:
        assert indices_path is not None and index_topk is not None
        row_bytes = (index_topk + 1) * 4
        indices_u32_bytes = batch_tokens * row_bytes
        try:
            actual_indices_bytes = indices_path.stat().st_size
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to stat indices u32 file {indices_path}: {exc}"
            ) from exc
        if actual_indices_bytes != indices_u32_bytes:
            raise PrefillExecuteError(
                f"indices u32 file has {actual_indices_bytes} bytes, expected "
                f"{indices_u32_bytes}"
            )
        cache_read_bytes = 0
        try:
            with indices_path.open("rb") as indices_file:
                for token_index in range(batch_tokens):
                    row = indices_file.read(row_bytes)
                    if len(row) != row_bytes:
                        raise PrefillExecuteError("failed to read full indices u32 row")
                    count = struct.unpack_from("<I", row, 0)[0]
                    if count == 0 or count > index_topk:
                        raise PrefillExecuteError(
                            f"indices row {token_index} has invalid count {count}"
                        )
                    for idx in range(count):
                        position = struct.unpack_from("<I", row, 4 + idx * 4)[0]
                        if position >= context_length:
                            raise PrefillExecuteError(
                                f"indices row {token_index} position {position} "
                                f"exceeds context_length {context_length}"
                            )
                    cache_read_bytes += count * segment.width * segment.dtype_bytes
        except OSError as exc:
            raise PrefillExecuteError(
                f"failed to read indices u32 file {indices_path}: {exc}"
            ) from exc
    else:
        cache_read_bytes = context_length * segment.width * segment.dtype_bytes
    if cache_read_bytes > max_cache_read_bytes:
        raise PrefillExecuteError(
            f"cache read {cache_read_bytes} bytes exceeds limit {max_cache_read_bytes}"
        )

    q_nope_path = Path(q_nope_f32_path)
    q_rope_path = Path(q_rope_f32_path)
    q_nope_bytes = batch_tokens * num_heads * qk_nope_dim * 4
    q_rope_bytes = batch_tokens * num_heads * rope_dim * 4
    for path, expected in ((q_nope_path, q_nope_bytes), (q_rope_path, q_rope_bytes)):
        try:
            actual = path.stat().st_size
        except OSError as exc:
            raise PrefillExecuteError(f"failed to stat query file {path}: {exc}") from exc
        if actual != expected:
            raise PrefillExecuteError(
                f"query file {path} has {actual} bytes, expected {expected}"
            )

    cache_f32_bytes = (
        batch_tokens * int(index_topk or 0) * cache_width * 4
        if indexed
        else context_length * cache_width * 4
    )
    kv_b_f32_bytes = kv_b_out * kv_b_in * 4
    output_bytes = batch_tokens * num_heads * v_head_dim * 4
    mla_key_cache_bytes = (
        context_length * num_heads * qk_nope_dim * 4
        if mla_key_cache and not indexed
        else 0
    )
    value_cache_batch_allowed = (
        batch_tokens > 1 or _mla_value_cache_singleton_enabled_by_env()
    )
    mla_value_cache_candidate_bytes = (
        context_length * num_heads * v_head_dim * 4
        if mla_value_cache and not indexed and value_cache_batch_allowed
        else 0
    )
    estimated_peak = (
        cache_read_bytes
        + cache_f32_bytes
        + kv_b_matrix_bytes
        + kv_b_source_f32_bytes
        + kv_b_f32_bytes
        + q_nope_bytes
        + q_rope_bytes
        + indices_u32_bytes
        + output_bytes
        + mla_key_cache_bytes
    )
    max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    if estimated_peak > max_scratch_bytes:
        raise PrefillExecuteError(
            f"estimated peak {estimated_peak} bytes exceeds scratch limit "
            f"{max_scratch_bytes}"
        )
    mla_value_cache_bytes = 0
    if mla_value_cache_candidate_bytes:
        value_cache_peak = estimated_peak + mla_value_cache_candidate_bytes
        if value_cache_peak <= max_scratch_bytes:
            estimated_peak = value_cache_peak
            mla_value_cache_bytes = mla_value_cache_candidate_bytes

    actual_scale = (
        attention_scale
        if attention_scale is not None
        else 1.0 / math.sqrt(float(qk_nope_dim + rope_dim))
    )
    output_path = Path(output_f32_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(runner),
        "--resident-layout",
        str(resident_layout),
        "--cache-layout",
        str(cache_layout_p),
        "--cache-file",
        str(cache_file_p),
        "--layer",
        str(layer),
        "--run-mla-attention-indexed-batch" if indexed else "--run-mla-attention-batch",
        "--q-nope-f32",
        str(q_nope_path),
        "--q-rope-f32",
        str(q_rope_path),
    ]
    if indexed:
        assert indices_path is not None and index_topk is not None
        cmd.extend(
            [
                "--indices-u32",
                str(indices_path),
                "--index-topk",
                str(index_topk),
            ]
        )
    cmd.extend(
        [
            "--context-length",
            str(context_length),
            "--batch-tokens",
            str(batch_tokens),
            "--num-heads",
            str(num_heads),
            "--qk-nope-dim",
            str(qk_nope_dim),
            "--rope-dim",
            str(rope_dim),
            "--v-head-dim",
            str(v_head_dim),
            "--output-f32",
            str(output_path),
            "--kv-lora-dim",
            str(resolved_kv_lora),
            "--cache-position-offset",
            str(cache_position_offset),
            "--rope-theta",
            f"{rope_theta:.9g}",
            "--max-cache-file-mib",
            str(max_cache_file_mib),
            "--max-cache-read-mib",
            str(max_cache_read_mib),
            "--max-resident-matrix-mib",
            str(max_resident_matrix_mib),
            "--max-runner-scratch-mib",
            str(max_runner_scratch_mib),
        ]
    )
    if not indexed:
        cmd.extend(["--start-position", str(start_position)])
    if attention_scale is not None:
        cmd.extend(["--attention-scale", f"{attention_scale:.9g}"])
    if rope_interleave:
        cmd.append("--rope-interleave")
    if mla_kv_b_cache_path is not None:
        cmd.extend(["--mla-kv-b-cache-dir", str(mla_kv_b_cache_path)])
    disable_value_cache = (
        not mla_value_cache
        or (
            mla_value_cache_candidate_bytes > 0
            and mla_value_cache_bytes == 0
        )
    )
    runner_env = None
    if mla_key_cache or disable_value_cache:
        runner_env = os.environ.copy()
        if mla_key_cache:
            runner_env[MLA_KEY_CACHE_ENV] = "1"
        if disable_value_cache:
            runner_env[MLA_VALUE_CACHE_DISABLE_ENV] = "1"
    runner_stdout: str
    recorded_command: tuple[str, ...] = tuple(cmd)
    if (
        mla_attention_server_session is not None
        and mla_attention_server_session.is_compatible(
            mla_key_cache=mla_key_cache,
            disable_value_cache=disable_value_cache,
        )
    ):
        runner_stdout = mla_attention_server_session.submit_batch(
            resident_layout_path=resident_layout,
            cache_layout_path=cache_layout_p,
            cache_file_path=cache_file_p,
            layer=layer,
            q_nope_f32_path=q_nope_path,
            q_rope_f32_path=q_rope_path,
            output_f32_path=output_path,
            context_length=context_length,
            start_position=start_position,
            batch_tokens=batch_tokens,
            indices_u32_path=indices_path,
            index_topk=index_topk,
            num_heads=num_heads,
            qk_nope_dim=qk_nope_dim,
            rope_dim=rope_dim,
            v_head_dim=v_head_dim,
            mla_kv_b_cache_dir=mla_kv_b_cache_path,
            kv_lora_dim=resolved_kv_lora,
            cache_position_offset=cache_position_offset,
            attention_scale=attention_scale,
            rope_theta=rope_theta,
            rope_interleave=rope_interleave,
            max_cache_file_mib=int(max_cache_file_mib),
            max_cache_read_mib=int(max_cache_read_mib),
            max_resident_matrix_mib=int(max_resident_matrix_mib),
            max_runner_scratch_mib=int(max_runner_scratch_mib),
        )
        recorded_command = (str(runner), "--run-mla-attention-batch-server-jsonl")
    else:
        completed = _run_command(cmd, echo_output=echo_runner_output, env=runner_env)
        runner_stdout = completed.stdout
    runner_timings = _parse_runner_timing_seconds(runner_stdout)
    try:
        actual_output_bytes = output_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat attention output {output_path}: {exc}") from exc
    if actual_output_bytes != output_bytes:
        raise PrefillExecuteError(
            f"attention output has {actual_output_bytes} bytes, expected {output_bytes}"
        )

    return PrefillMLAAttentionBatchResult(
        runner_path=runner,
        resident_layout_path=resident_layout,
        cache_layout_path=cache_layout_p,
        cache_file_path=cache_file_p,
        q_nope_path=q_nope_path,
        q_rope_path=q_rope_path,
        indices_u32_path=indices_path,
        output_path=output_path,
        mla_kv_b_cache_dir=mla_kv_b_cache_path,
        layer=layer,
        context_length=context_length,
        start_position=start_position,
        batch_tokens=batch_tokens,
        num_heads=num_heads,
        kv_lora_dim=resolved_kv_lora,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
        cache_position_offset=cache_position_offset,
        indexed=indexed,
        index_topk=index_topk,
        attention_scale=actual_scale,
        rope_theta=rope_theta,
        rope_interleave=rope_interleave,
        attention_value_source=attention_value_source,
        q_nope_bytes=q_nope_bytes,
        q_rope_bytes=q_rope_bytes,
        indices_u32_bytes=indices_u32_bytes,
        output_bytes=output_bytes,
        cache_read_bytes=cache_read_bytes,
        cache_f32_bytes=cache_f32_bytes,
        kv_b_matrix_bytes=kv_b_matrix_bytes,
        kv_b_f32_bytes=kv_b_f32_bytes,
        mla_key_cache=mla_key_cache,
        mla_key_cache_bytes=mla_key_cache_bytes,
        mla_value_cache=mla_value_cache_bytes > 0,
        mla_value_cache_bytes=mla_value_cache_bytes,
        estimated_peak_bytes=estimated_peak,
        command=recorded_command,
        mla_timing_input_elapsed_seconds=runner_timings.get("input"),
        mla_timing_cache_read_elapsed_seconds=runner_timings.get("cache_read"),
        mla_timing_value_read_elapsed_seconds=runner_timings.get("value_read"),
        mla_timing_metal_setup_elapsed_seconds=runner_timings.get("metal_setup"),
        mla_timing_kernel_elapsed_seconds=runner_timings.get("kernel"),
        mla_timing_write_elapsed_seconds=runner_timings.get("write"),
        mla_timing_total_elapsed_seconds=runner_timings.get("total"),
    )


def run_prefill_attention_output_batch(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    attn_value_f32_path: str | Path,
    residual_f32_path: str | Path,
    output_f32_path: str | Path,
    batch_tokens: int,
    o_proj_suffix: str = ".self_attn.o_proj.weight",
    projection_f32_path: str | Path | None = None,
    max_resident_matrix_mib: int = 512,
    max_runner_scratch_mib: int = 4096,
    prefill_linear_backend: str = "custom-metal",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    echo_runner_output: bool = True,
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None,
    attention_output_server_session: AttentionOutputBatchServerSession | None = None,
) -> PrefillAttentionOutputBatchResult:
    if batch_tokens <= 0:
        raise PrefillExecuteError("batch_tokens must be positive")
    if max_resident_matrix_mib <= 0 or max_runner_scratch_mib <= 0:
        raise PrefillExecuteError("matrix and scratch limits must be positive")

    output_path = Path(output_f32_path)
    projection_path = (
        Path(projection_f32_path)
        if projection_f32_path is not None
        else output_path.with_name(output_path.stem + ".o_proj.f32")
    )
    if (
        batch_tokens == 1
        and Path(runner_path).name == "largerlm-runner"
        and o_proj_suffix == ".self_attn.o_proj.weight"
        and prefill_linear_backend in {"custom-metal", "auto"}
    ):
        runner = Path(runner_path)
        if not runner.exists():
            raise PrefillExecuteError(f"runner not found: {runner}")
        layout_path = Path(resident_layout_path)
        layout = _load_json(layout_path)
        weight_file = layout.get("weight_file")
        if not isinstance(weight_file, str):
            raise PrefillExecuteError("resident layout missing weight_file")
        weight_path = layout_path.parent / weight_file
        if not weight_path.exists():
            raise PrefillExecuteError(f"resident weight file not found: {weight_path}")
        matrix = _find_layer_matrix(layout, layer=layer, suffix=o_proj_suffix)
        out_dim, in_dim, matrix_bytes, dtype = _resident_matrix_metadata(
            layout=layout,
            weight_path=weight_path,
            matrix=matrix,
            label=str(matrix.get("name") or "attention o_proj"),
        )
        max_resident_matrix_mib = _positive_integer_mib(
            "max_resident_matrix_mib",
            max_resident_matrix_mib,
        )
        max_runner_scratch_mib = _positive_integer_mib(
            "max_runner_scratch_mib",
            max_runner_scratch_mib,
        )
        max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
        if matrix_bytes > max_matrix_bytes:
            raise PrefillExecuteError(
                f"attention o_proj has {matrix_bytes} bytes, exceeds limit {max_matrix_bytes}"
            )
        attn_value_path = Path(attn_value_f32_path)
        residual_path = Path(residual_f32_path)
        input_bytes = in_dim * 4
        output_bytes = out_dim * 4
        _stat_expected_file(attn_value_path, input_bytes, "attention value")
        _stat_expected_file(residual_path, output_bytes, "attention residual")
        matrix_scratch = _resident_linear_matrix_scratch(
            matrix_bytes=matrix_bytes,
            dtype=dtype,
            in_dim=in_dim,
            out_dim=out_dim,
            backend="custom-metal",
        )
        runner_peak = matrix_scratch.matrix_scratch_bytes + input_bytes + 2 * output_bytes
        max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
        if runner_peak > max_scratch_bytes:
            raise PrefillExecuteError(
                f"estimated peak {runner_peak} bytes exceeds scratch limit "
                f"{max_scratch_bytes}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(runner),
            "--resident-layout",
            str(layout_path),
            "--layer",
            str(layer),
            "--run-attn-output",
            "--input-f32",
            str(attn_value_path),
            "--residual-f32",
            str(residual_path),
            "--output-f32",
            str(output_path),
            "--projection-f32",
            str(projection_path),
            "--max-resident-matrix-mib",
            str(max_resident_matrix_mib),
            "--max-runner-scratch-mib",
            str(max_runner_scratch_mib),
        ]
        started = time.perf_counter()
        completed = _run_command(cmd, echo_output=echo_runner_output)
        elapsed_seconds = time.perf_counter() - started
        runner_timings = _parse_runner_timing_seconds(completed.stdout)
        _stat_expected_file(projection_path, output_bytes, "attention projection")
        _stat_expected_file(output_path, output_bytes, "attention output")
        o_proj = _resident_linear_result_metadata(
            runner=runner,
            layout_path=layout_path,
            matrix=matrix,
            tensor_suffix=o_proj_suffix,
            input_path=attn_value_path,
            output_path=projection_path,
            layer=layer,
            batch_tokens=1,
            in_dim=in_dim,
            out_dim=out_dim,
            matrix_bytes=matrix_bytes,
            dtype=dtype,
            backend="fused-metal",
            command=tuple(cmd),
        )
        o_proj = replace(
            o_proj,
            elapsed_seconds=elapsed_seconds,
            estimated_peak_bytes=runner_peak,
            runner_backend_elapsed_seconds=runner_timings.get("backend"),
        )
        return PrefillAttentionOutputBatchResult(
            runner_path=runner,
            resident_layout_path=layout_path,
            attn_value_path=attn_value_path,
            residual_path=residual_path,
            projection_path=projection_path,
            output_path=output_path,
            layer=layer,
            batch_tokens=1,
            attn_value_dim=in_dim,
            hidden_dim=out_dim,
            attn_value_bytes=input_bytes,
            residual_bytes=output_bytes,
            projection_bytes=output_bytes,
            output_bytes=output_bytes,
            residual_add_peak_bytes=2 * output_bytes,
            estimated_peak_bytes=runner_peak,
            o_proj=o_proj,
        )
    if (
        batch_tokens > 1
        and Path(runner_path).name == "largerlm-runner"
        and o_proj_suffix == ".self_attn.o_proj.weight"
        and prefill_linear_backend in {"custom-metal", "auto"}
        and _batch_fused_attention_output_enabled()
    ):
        runner = Path(runner_path)
        if not runner.exists():
            raise PrefillExecuteError(f"runner not found: {runner}")
        layout_path = Path(resident_layout_path)
        layout = _load_json(layout_path)
        weight_file = layout.get("weight_file")
        if not isinstance(weight_file, str):
            raise PrefillExecuteError("resident layout missing weight_file")
        weight_path = layout_path.parent / weight_file
        if not weight_path.exists():
            raise PrefillExecuteError(f"resident weight file not found: {weight_path}")
        matrix = _find_layer_matrix(layout, layer=layer, suffix=o_proj_suffix)
        out_dim, in_dim, matrix_bytes, dtype = _resident_matrix_metadata(
            layout=layout,
            weight_path=weight_path,
            matrix=matrix,
            label=str(matrix.get("name") or "attention o_proj"),
        )
        max_resident_matrix_mib = _positive_integer_mib(
            "max_resident_matrix_mib",
            max_resident_matrix_mib,
        )
        max_runner_scratch_mib = _positive_integer_mib(
            "max_runner_scratch_mib",
            max_runner_scratch_mib,
        )
        max_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
        if matrix_bytes > max_matrix_bytes:
            raise PrefillExecuteError(
                f"attention o_proj has {matrix_bytes} bytes, exceeds limit {max_matrix_bytes}"
            )
        attn_value_path = Path(attn_value_f32_path)
        residual_path = Path(residual_f32_path)
        input_bytes = batch_tokens * in_dim * 4
        output_bytes = batch_tokens * out_dim * 4
        _stat_expected_file(attn_value_path, input_bytes, "attention value")
        _stat_expected_file(residual_path, output_bytes, "attention residual")
        matrix_scratch = _resident_linear_matrix_scratch(
            matrix_bytes=matrix_bytes,
            dtype=dtype,
            in_dim=in_dim,
            out_dim=out_dim,
            backend="custom-metal",
        )
        runner_peak = matrix_scratch.matrix_scratch_bytes + input_bytes + 2 * output_bytes
        max_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
        if runner_peak > max_scratch_bytes:
            raise PrefillExecuteError(
                f"estimated peak {runner_peak} bytes exceeds scratch limit "
                f"{max_scratch_bytes}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        projection_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(runner),
            "--resident-layout",
            str(layout_path),
            "--layer",
            str(layer),
            "--run-attn-output-batch",
            "--input-f32",
            str(attn_value_path),
            "--residual-f32",
            str(residual_path),
            "--batch-tokens",
            str(batch_tokens),
            "--output-f32",
            str(output_path),
            "--projection-f32",
            str(projection_path),
            "--max-resident-matrix-mib",
            str(max_resident_matrix_mib),
            "--max-runner-scratch-mib",
            str(max_runner_scratch_mib),
        ]
        recorded_command: tuple[str, ...] = tuple(cmd)
        started = time.perf_counter()
        if attention_output_server_session is not None:
            runner_stdout = attention_output_server_session.submit_batch(
                resident_layout_path=layout_path,
                layer=layer,
                input_f32_path=attn_value_path,
                residual_f32_path=residual_path,
                output_f32_path=output_path,
                projection_f32_path=projection_path,
                batch_tokens=batch_tokens,
                max_resident_matrix_mib=max_resident_matrix_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
            )
            recorded_command = (str(runner), "--run-attn-output-batch-server-jsonl")
        else:
            completed = _run_command(cmd, echo_output=echo_runner_output)
            runner_stdout = completed.stdout
        elapsed_seconds = time.perf_counter() - started
        runner_timings = _parse_runner_timing_seconds(runner_stdout)
        _stat_expected_file(projection_path, output_bytes, "attention projection")
        _stat_expected_file(output_path, output_bytes, "attention output")
        o_proj = _resident_linear_result_metadata(
            runner=runner,
            layout_path=layout_path,
            matrix=matrix,
            tensor_suffix=o_proj_suffix,
            input_path=attn_value_path,
            output_path=projection_path,
            layer=layer,
            batch_tokens=batch_tokens,
            in_dim=in_dim,
            out_dim=out_dim,
            matrix_bytes=matrix_bytes,
            dtype=dtype,
            backend="fused-metal",
            command=recorded_command,
        )
        o_proj = replace(
            o_proj,
            elapsed_seconds=elapsed_seconds,
            estimated_peak_bytes=runner_peak,
            runner_backend_elapsed_seconds=runner_timings.get("backend"),
        )
        return PrefillAttentionOutputBatchResult(
            runner_path=runner,
            resident_layout_path=layout_path,
            attn_value_path=attn_value_path,
            residual_path=residual_path,
            projection_path=projection_path,
            output_path=output_path,
            layer=layer,
            batch_tokens=batch_tokens,
            attn_value_dim=in_dim,
            hidden_dim=out_dim,
            attn_value_bytes=input_bytes,
            residual_bytes=output_bytes,
            projection_bytes=output_bytes,
            output_bytes=output_bytes,
            residual_add_peak_bytes=2 * output_bytes,
            estimated_peak_bytes=runner_peak,
            o_proj=o_proj,
        )
    o_proj = run_resident_batch_linear(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        tensor_suffix=o_proj_suffix,
        input_f32_path=attn_value_f32_path,
        output_f32_path=projection_path,
        batch_tokens=batch_tokens,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
    )
    residual_path = Path(residual_f32_path)
    output_bytes, add_peak = _add_f32_batches_streaming(
        lhs_path=projection_path,
        rhs_path=residual_path,
        output_path=output_path,
        batch_tokens=batch_tokens,
        dim=o_proj.out_dim,
    )
    try:
        residual_bytes = residual_path.stat().st_size
        actual_output_bytes = output_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat attention output files: {exc}") from exc
    if actual_output_bytes != output_bytes:
        raise PrefillExecuteError(
            f"attention output bytes {actual_output_bytes} do not match expected {output_bytes}"
        )
    estimated_peak = max(o_proj.estimated_peak_bytes, add_peak)
    return PrefillAttentionOutputBatchResult(
        runner_path=o_proj.runner_path,
        resident_layout_path=o_proj.resident_layout_path,
        attn_value_path=o_proj.input_path,
        residual_path=residual_path,
        projection_path=projection_path,
        output_path=output_path,
        layer=layer,
        batch_tokens=batch_tokens,
        attn_value_dim=o_proj.in_dim,
        hidden_dim=o_proj.out_dim,
        attn_value_bytes=o_proj.input_bytes,
        residual_bytes=residual_bytes,
        projection_bytes=o_proj.output_bytes,
        output_bytes=output_bytes,
        residual_add_peak_bytes=add_peak,
        estimated_peak_bytes=estimated_peak,
        o_proj=o_proj,
    )


def run_prefill_attention_block_batch(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    cache_layout_path: str | Path,
    cache_file_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_dir: str | Path,
    output_f32_path: str | Path,
    start_position: int,
    batch_tokens: int,
    num_heads: int,
    qk_nope_dim: int,
    rope_dim: int,
    v_head_dim: int,
    mla_kv_b_cache_dir: str | Path | None = None,
    mla_key_cache: bool = False,
    mla_value_cache: bool = True,
    context_length: int | None = None,
    kv_lora_dim: int | None = None,
    cache_position_offset: int = 0,
    attention_scale: float | None = None,
    rope_theta: float = 10000.0,
    rope_interleave: bool = False,
    dsa_indexer_mode: str = "none",
    dsa_prev_indices_u32_path: str | Path | None = None,
    dsa_index_topk: int | None = None,
    dsa_index_n_heads: int | None = None,
    dsa_qk_rope_dim: int | None = None,
    dsa_rope_interleave: bool = False,
    dsa_layer_norm_eps: float = 1e-6,
    dsa_indices_u32_path: str | Path | None = None,
    write_dsa_future_cache: bool = True,
    rms_norm_eps: float = 1e-5,
    max_cache_file_mib: int = 32768,
    max_cache_write_mib: int = 4096,
    max_cache_read_mib: int = 256,
    max_resident_matrix_mib: int = 512,
    max_runner_scratch_mib: int = 4096,
    prefill_linear_backend: str = "custom-metal",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    echo_runner_output: bool = True,
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None,
    attention_projection_server_session: AttentionProjectionsServerSession | None = None,
    attention_output_server_session: AttentionOutputBatchServerSession | None = None,
    rope_split_server_session: RopeSplitBatchServerSession | None = None,
    mla_attention_server_session: MLAAttentionBatchServerSession | None = None,
    rmsnorm_server_session: ResidentBatchRMSNormServerSession | None = None,
) -> PrefillAttentionBlockBatchResult:
    if type(write_dsa_future_cache) is not bool:
        raise PrefillExecuteError("write_dsa_future_cache must be a boolean")
    if type(mla_key_cache) is not bool:
        raise PrefillExecuteError("mla_key_cache must be a boolean")
    if type(mla_value_cache) is not bool:
        raise PrefillExecuteError("mla_value_cache must be a boolean")
    if start_position < 0 or batch_tokens <= 0:
        raise PrefillExecuteError("start_position and batch_tokens are invalid")
    if num_heads <= 0 or qk_nope_dim <= 0 or rope_dim <= 0 or v_head_dim <= 0:
        raise PrefillExecuteError("attention dimensions must be positive")
    resolved_context_length = (
        start_position + batch_tokens if context_length is None else context_length
    )
    if resolved_context_length <= 0:
        raise PrefillExecuteError("context_length must be positive")
    if dsa_indexer_mode not in {"none", "full", "shared"}:
        raise PrefillExecuteError("dsa_indexer_mode must be none, full, or shared")
    if dsa_indexer_mode == "full":
        if dsa_index_topk is None or dsa_index_n_heads is None:
            raise PrefillExecuteError("full DSA indexer requires index_topk and index_n_heads")
        if dsa_index_topk <= 0 or dsa_index_n_heads <= 0:
            raise PrefillExecuteError("DSA index_topk and index_n_heads must be positive")
        if dsa_layer_norm_eps <= 0:
            raise PrefillExecuteError("dsa_layer_norm_eps must be positive")
    elif dsa_indexer_mode == "shared":
        if dsa_index_topk is None or dsa_index_topk <= 0:
            raise PrefillExecuteError("shared DSA indexer requires positive index_topk")
        if (
            dsa_prev_indices_u32_path is None
            and resolved_context_length > int(dsa_index_topk)
        ):
            raise PrefillExecuteError("shared DSA indexer requires previous indices")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    projections_dir = out_dir / "projections"
    rope_dir = out_dir / "rope"
    attn_value_path = out_dir / "attn_value.f32"
    attn_projection_path = out_dir / "attn_output.o_proj.f32"
    dsa_topk_path = (
        Path(dsa_indices_u32_path)
        if dsa_indices_u32_path is not None
        else out_dir / "dsa_topk.u32"
    )

    projections_started = time.perf_counter()
    projections = run_prefill_attention_projection_batch(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        input_f32_path=input_f32_path,
        output_dir=projections_dir,
        batch_tokens=batch_tokens,
        rms_norm_eps=rms_norm_eps,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
        attention_projection_server_session=attention_projection_server_session,
        rmsnorm_server_session=rmsnorm_server_session,
    )
    projections_elapsed = time.perf_counter() - projections_started
    resolved_kv_lora = kv_lora_dim if kv_lora_dim is not None else projections.kv_lora_dim
    if resolved_kv_lora != projections.kv_lora_dim:
        raise PrefillExecuteError(
            f"kv_lora_dim {resolved_kv_lora} does not match projection split "
            f"{projections.kv_lora_dim}"
        )

    cache_write_started = time.perf_counter()
    cache_write = write_prefill_kv_cache_batch(
        cache_layout_path=cache_layout_path,
        cache_file_path=cache_file_path,
        layer=layer,
        input_f32_path=projections.prefix.kv_a_proj_with_mqa.output_path,
        start_position=start_position,
        batch_tokens=batch_tokens,
        max_cache_file_mib=max_cache_file_mib,
        max_cache_write_mib=max_cache_write_mib,
    )
    cache_write_elapsed = time.perf_counter() - cache_write_started
    cache_write_peak = cache_write.estimated_peak_bytes

    rope_started = time.perf_counter()
    rope = run_prefill_rope_batch(
        runner_path=runner_path,
        q_b_f32_path=projections.q_b_proj.output_path,
        k_rope_f32_path=projections.kv_a_rope_path,
        output_dir=rope_dir,
        batch_tokens=batch_tokens,
        num_heads=num_heads,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
        start_position=start_position,
        rope_theta=rope_theta,
        rope_interleave=rope_interleave,
        max_runner_scratch_mib=max_runner_scratch_mib,
        echo_runner_output=echo_runner_output,
        rope_split_server_session=rope_split_server_session,
    )
    rope_elapsed = time.perf_counter() - rope_started
    dsa_indexer: DSAIndexerBatchResult | None = None
    active_dsa_indices: Path | None = None
    dsa_indexer_elapsed = 0.0
    dsa_indexer_peak = 0
    dsa_indices_cover_visible_context = (
        dsa_index_topk is not None
        and resolved_context_length <= int(dsa_index_topk)
    )
    if dsa_indexer_mode == "full":
        dsa_qk_dim = rope_dim if dsa_qk_rope_dim is None else dsa_qk_rope_dim
        dsa_indexer_started = time.perf_counter()
        visible_dsa_rows = (
            _visible_dsa_topk_rows(
                start_position=start_position,
                batch_tokens=batch_tokens,
                context_length=resolved_context_length,
                index_topk=int(dsa_index_topk or 0),
            )
            if not write_dsa_future_cache
            else None
        )
        if visible_dsa_rows is not None:
            _write_visible_dsa_topk_u32(
                dsa_topk_path,
                rows=visible_dsa_rows,
                index_topk=int(dsa_index_topk or 0),
            )
            active_dsa_indices = dsa_topk_path
            dsa_indexer_elapsed = time.perf_counter() - dsa_indexer_started
        elif dsa_indices_cover_visible_context:
            try:
                dsa_cache_write = write_dsa_index_cache_batch(
                    resident_layout_path=resident_layout_path,
                    cache_layout_path=cache_layout_path,
                    cache_file_path=cache_file_path,
                    layer=layer,
                    hidden_f32_path=input_f32_path,
                    start_position=start_position,
                    batch_tokens=batch_tokens,
                    qk_rope_dim=dsa_qk_dim,
                    rope_theta=rope_theta,
                    rope_interleave=dsa_rope_interleave,
                    layer_norm_eps=dsa_layer_norm_eps,
                    max_resident_matrix_mib=max_resident_matrix_mib,
                    max_cache_file_mib=max_cache_file_mib,
                    max_cache_write_mib=max_cache_write_mib,
                    max_runner_scratch_mib=max_runner_scratch_mib,
                )
                dsa_indexer_peak = dsa_cache_write.estimated_peak_bytes
                dsa_indexer_elapsed = time.perf_counter() - dsa_indexer_started
            except DSAIndexerError as exc:
                raise PrefillExecuteError(str(exc)) from exc
        else:
            try:
                dsa_indexer = run_dsa_indexer_batch(
                    resident_layout_path=resident_layout_path,
                    cache_layout_path=cache_layout_path,
                    cache_file_path=cache_file_path,
                    layer=layer,
                    hidden_f32_path=input_f32_path,
                    q_resid_f32_path=projections.q_a_layernorm.output_path,
                    output_indices_path=None,
                    output_indices_u32_path=dsa_topk_path,
                    start_position=start_position,
                    batch_tokens=batch_tokens,
                    context_length=resolved_context_length,
                    index_topk=int(dsa_index_topk or 0),
                    index_n_heads=int(dsa_index_n_heads or 0),
                    qk_rope_dim=dsa_qk_dim,
                    rope_theta=rope_theta,
                    rope_interleave=dsa_rope_interleave,
                    layer_norm_eps=dsa_layer_norm_eps,
                    max_resident_matrix_mib=max_resident_matrix_mib,
                    max_cache_file_mib=max_cache_file_mib,
                    max_cache_write_mib=max_cache_write_mib,
                    max_cache_read_mib=max_cache_read_mib,
                    max_runner_scratch_mib=max_runner_scratch_mib,
                    collect_topk_indices=False,
                )
                dsa_indexer_elapsed = time.perf_counter() - dsa_indexer_started
            except DSAIndexerError as exc:
                raise PrefillExecuteError(str(exc)) from exc
            dsa_indexer_peak = max(
                dsa_indexer.cache_write.estimated_peak_bytes,
                dsa_indexer.topk.estimated_peak_bytes,
            )
            active_dsa_indices = dsa_indexer.topk.output_indices_u32_path
    elif dsa_indexer_mode == "shared":
        active_dsa_indices = (
            Path(dsa_prev_indices_u32_path)
            if dsa_prev_indices_u32_path is not None
            else None
        )

    mla_attention_started = time.perf_counter()
    mla_indices = (
        None
        if (
            (batch_tokens == 1 and resolved_context_length == 1)
            or dsa_indices_cover_visible_context
        )
        else active_dsa_indices
    )
    mla_index_topk = (
        None
        if mla_indices is None
        else dsa_index_topk
    )
    if (
        batch_tokens == 1
        and start_position == 0
        and resolved_context_length == 1
        and projections.kv_b_proj is not None
        and Path(runner_path).name == "largerlm-runner"
    ):
        expected_kv_b_dim = num_heads * (qk_nope_dim + v_head_dim)
        if projections.kv_b_dim != expected_kv_b_dim:
            raise PrefillExecuteError(
                f"singleton kv_b dim {projections.kv_b_dim} does not match "
                f"expected {expected_kv_b_dim}"
            )
        value_output_bytes, value_peak = _write_singleton_mla_value_from_kv_b(
            kv_b_output_path=projections.kv_b_proj.output_path,
            output_path=attn_value_path,
            num_heads=num_heads,
            qk_nope_dim=qk_nope_dim,
            v_head_dim=v_head_dim,
        )
        indices_u32_bytes = (
            (int(dsa_index_topk or 0) + 1) * 4
            if mla_indices is not None
            else 0
        )
        actual_scale = (
            attention_scale
            if attention_scale is not None
            else 1.0 / math.sqrt(float(qk_nope_dim + rope_dim))
        )
        singleton_command = (
            "python-singleton-mla-value",
            "--kv-b-f32",
            str(projections.kv_b_proj.output_path),
            "--output-f32",
            str(attn_value_path),
            "--num-heads",
            str(num_heads),
            "--qk-nope-dim",
            str(qk_nope_dim),
            "--v-head-dim",
            str(v_head_dim),
        )
        mla_attention = PrefillMLAAttentionBatchResult(
            runner_path=projections.runner_path,
            resident_layout_path=projections.resident_layout_path,
            cache_layout_path=Path(cache_layout_path),
            cache_file_path=Path(cache_file_path),
            q_nope_path=rope.q_nope_path,
            q_rope_path=rope.q_rope_rotated_path,
            indices_u32_path=mla_indices,
            output_path=attn_value_path,
            mla_kv_b_cache_dir=None,
            layer=layer,
            context_length=resolved_context_length,
            start_position=start_position,
            batch_tokens=batch_tokens,
            num_heads=num_heads,
            kv_lora_dim=resolved_kv_lora,
            qk_nope_dim=qk_nope_dim,
            rope_dim=rope_dim,
            v_head_dim=v_head_dim,
            cache_position_offset=cache_position_offset,
            indexed=mla_indices is not None,
            index_topk=mla_index_topk,
            attention_scale=actual_scale,
            rope_theta=rope_theta,
            rope_interleave=rope_interleave,
            attention_value_source="kv_b_proj-singleton",
            q_nope_bytes=rope.q_nope_bytes,
            q_rope_bytes=rope.q_rope_bytes,
            indices_u32_bytes=indices_u32_bytes,
            output_bytes=value_output_bytes,
            cache_read_bytes=0,
            cache_f32_bytes=0,
            kv_b_matrix_bytes=0,
            kv_b_f32_bytes=0,
            mla_key_cache=False,
            mla_key_cache_bytes=0,
            mla_value_cache=False,
            mla_value_cache_bytes=0,
            estimated_peak_bytes=max(value_peak, indices_u32_bytes),
            command=singleton_command,
        )
    else:
        effective_mla_key_cache = (
            mla_key_cache and mla_indices is None
        )
        mla_attention = run_prefill_mla_attention_batch(
            runner_path=runner_path,
            resident_layout_path=resident_layout_path,
            cache_layout_path=cache_layout_path,
            cache_file_path=cache_file_path,
            layer=layer,
            q_nope_f32_path=rope.q_nope_path,
            q_rope_f32_path=rope.q_rope_rotated_path,
            indices_u32_path=mla_indices,
            output_f32_path=attn_value_path,
            context_length=resolved_context_length,
            start_position=start_position,
            batch_tokens=batch_tokens,
            index_topk=mla_index_topk,
            num_heads=num_heads,
            qk_nope_dim=qk_nope_dim,
            rope_dim=rope_dim,
            v_head_dim=v_head_dim,
            mla_kv_b_cache_dir=mla_kv_b_cache_dir,
            mla_key_cache=effective_mla_key_cache,
            mla_value_cache=mla_value_cache,
            kv_lora_dim=resolved_kv_lora,
            cache_position_offset=cache_position_offset,
            attention_scale=attention_scale,
            rope_theta=rope_theta,
            rope_interleave=rope_interleave,
            max_cache_file_mib=max_cache_file_mib,
            max_cache_read_mib=max_cache_read_mib,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            echo_runner_output=echo_runner_output,
            mla_attention_server_session=mla_attention_server_session,
        )
    mla_attention_elapsed = time.perf_counter() - mla_attention_started
    attention_output_started = time.perf_counter()
    attention_output = run_prefill_attention_output_batch(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        attn_value_f32_path=mla_attention.output_path,
        residual_f32_path=input_f32_path,
        output_f32_path=output_f32_path,
        batch_tokens=batch_tokens,
        projection_f32_path=attn_projection_path,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
        attention_output_server_session=attention_output_server_session,
    )
    attention_output_elapsed = time.perf_counter() - attention_output_started
    estimated_peak = max(
        projections.estimated_peak_bytes,
        cache_write_peak,
        rope.estimated_peak_bytes,
        dsa_indexer_peak,
        mla_attention.estimated_peak_bytes,
        attention_output.estimated_peak_bytes,
    )
    return PrefillAttentionBlockBatchResult(
        runner_path=projections.runner_path,
        resident_layout_path=projections.resident_layout_path,
        cache_layout_path=cache_write.cache_layout_path,
        cache_file_path=cache_write.cache_file_path,
        input_path=projections.input_path,
        output_dir=out_dir,
        output_path=attention_output.output_path,
        layer=layer,
        context_length=resolved_context_length,
        start_position=start_position,
        batch_tokens=batch_tokens,
        num_heads=num_heads,
        kv_lora_dim=resolved_kv_lora,
        qk_nope_dim=qk_nope_dim,
        rope_dim=rope_dim,
        v_head_dim=v_head_dim,
        hidden_dim=attention_output.hidden_dim,
        mla_key_cache=mla_attention.mla_key_cache,
        mla_key_cache_bytes=mla_attention.mla_key_cache_bytes,
        mla_value_cache=mla_attention.mla_value_cache,
        mla_value_cache_bytes=mla_attention.mla_value_cache_bytes,
        input_bytes=projections.input_bytes,
        output_bytes=attention_output.output_bytes,
        cache_write_peak_bytes=cache_write_peak,
        estimated_peak_bytes=estimated_peak,
        projections_elapsed_seconds=projections_elapsed,
        cache_write_elapsed_seconds=cache_write_elapsed,
        rope_elapsed_seconds=rope_elapsed,
        dsa_indexer_elapsed_seconds=dsa_indexer_elapsed,
        mla_attention_elapsed_seconds=mla_attention_elapsed,
        attention_output_elapsed_seconds=attention_output_elapsed,
        projections=projections,
        cache_write=cache_write,
        dsa_indexer=dsa_indexer,
        dsa_rope_interleave=dsa_rope_interleave,
        dsa_indices_u32_path=active_dsa_indices,
        rope=rope,
        mla_attention=mla_attention,
        attention_output=attention_output,
    )


def run_prefill_dense_mlp_block_batch(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_dir: str | Path,
    output_f32_path: str | Path,
    batch_tokens: int,
    norm_suffix: str = ".post_attention_layernorm.weight",
    gate_suffix: str = ".mlp.gate_proj.weight",
    up_suffix: str = ".mlp.up_proj.weight",
    down_suffix: str = ".mlp.down_proj.weight",
    rms_norm_eps: float = 1e-5,
    max_resident_matrix_mib: int = 512,
    max_runner_scratch_mib: int = 4096,
    prefill_linear_backend: str = "custom-metal",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    echo_runner_output: bool = True,
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None,
    rmsnorm_server_session: ResidentBatchRMSNormServerSession | None = None,
) -> PrefillDenseMLPBlockBatchResult:
    if batch_tokens <= 0:
        raise PrefillExecuteError("batch_tokens must be positive")

    if gate_suffix == ".mlp.gate_proj.weight":
        gate_suffix = _resolve_layer_matrix_suffix(
            resident_layout_path,
            layer=layer,
            suffixes=(
                ".mlp.gate_proj.weight",
                ".mlp.switch_mlp.gate_proj.weight",
                ".switch_mlp.gate_proj.weight",
            ),
        )
    if up_suffix == ".mlp.up_proj.weight":
        up_suffix = _resolve_layer_matrix_suffix(
            resident_layout_path,
            layer=layer,
            suffixes=(
                ".mlp.up_proj.weight",
                ".mlp.switch_mlp.up_proj.weight",
                ".switch_mlp.up_proj.weight",
            ),
        )
    if down_suffix == ".mlp.down_proj.weight":
        down_suffix = _resolve_layer_matrix_suffix(
            resident_layout_path,
            layer=layer,
            suffixes=(
                ".mlp.down_proj.weight",
                ".mlp.switch_mlp.down_proj.weight",
                ".switch_mlp.down_proj.weight",
            ),
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_output = out_dir / "post_attention_layernorm.f32"
    gate_output = out_dir / "gate_proj.f32"
    up_output = out_dir / "up_proj.f32"
    swiglu_output = out_dir / "swiglu.f32"
    down_output = out_dir / "down_proj.f32"

    norm = run_resident_batch_rmsnorm(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        norm_suffix=norm_suffix,
        input_f32_path=input_f32_path,
        output_f32_path=norm_output,
        batch_tokens=batch_tokens,
        rms_norm_eps=rms_norm_eps,
        max_runner_scratch_mib=max_runner_scratch_mib,
        echo_runner_output=echo_runner_output,
        rmsnorm_server_session=rmsnorm_server_session,
    )
    gate = run_resident_batch_linear(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        tensor_suffix=gate_suffix,
        input_f32_path=norm.output_path,
        output_f32_path=gate_output,
        batch_tokens=batch_tokens,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
    )
    up = run_resident_batch_linear(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        tensor_suffix=up_suffix,
        input_f32_path=norm.output_path,
        output_f32_path=up_output,
        batch_tokens=batch_tokens,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
    )
    if gate.in_dim != norm.hidden_dim or up.in_dim != norm.hidden_dim:
        raise PrefillExecuteError("dense MLP gate/up input dims do not match hidden dim")
    if gate.out_dim != up.out_dim:
        raise PrefillExecuteError(
            f"dense MLP gate dim {gate.out_dim} does not match up dim {up.out_dim}"
        )

    swiglu_bytes, swiglu_peak = _swiglu_f32_batches_streaming(
        gate_path=gate.output_path,
        up_path=up.output_path,
        output_path=swiglu_output,
        batch_tokens=batch_tokens,
        dim=gate.out_dim,
    )
    down = run_resident_batch_linear(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        tensor_suffix=down_suffix,
        input_f32_path=swiglu_output,
        output_f32_path=down_output,
        batch_tokens=batch_tokens,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
    )
    if down.in_dim != gate.out_dim or down.out_dim != norm.hidden_dim:
        raise PrefillExecuteError(
            f"dense MLP down shape [{down.out_dim},{down.in_dim}] does not match "
            f"[{norm.hidden_dim},{gate.out_dim}]"
        )
    output_bytes, add_peak = _add_f32_batches_streaming(
        lhs_path=down.output_path,
        rhs_path=norm.input_path,
        output_path=Path(output_f32_path),
        batch_tokens=batch_tokens,
        dim=norm.hidden_dim,
    )
    estimated_peak = max(
        norm.estimated_peak_bytes,
        gate.estimated_peak_bytes,
        up.estimated_peak_bytes,
        swiglu_peak,
        down.estimated_peak_bytes,
        add_peak,
    )
    return PrefillDenseMLPBlockBatchResult(
        runner_path=norm.runner_path,
        resident_layout_path=norm.resident_layout_path,
        input_path=norm.input_path,
        output_dir=out_dir,
        output_path=Path(output_f32_path),
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=norm.hidden_dim,
        intermediate_dim=gate.out_dim,
        input_bytes=norm.input_bytes,
        norm_output_bytes=norm.output_bytes,
        gate_output_bytes=gate.output_bytes,
        up_output_bytes=up.output_bytes,
        swiglu_output_bytes=swiglu_bytes,
        down_output_bytes=down.output_bytes,
        output_bytes=output_bytes,
        swiglu_peak_bytes=swiglu_peak,
        residual_add_peak_bytes=add_peak,
        estimated_peak_bytes=estimated_peak,
        post_attention_layernorm=norm,
        gate_proj=gate,
        up_proj=up,
        down_proj=down,
    )


def _resident_weight_path_from_layout(
    resident_layout_path: Path,
    layout: dict[str, Any],
) -> Path:
    weight_file = layout.get("weight_file")
    if not isinstance(weight_file, str):
        raise PrefillExecuteError("resident layout missing weight_file")
    return resident_layout_path.parent / weight_file


def _read_resident_vector_f32(
    *,
    resident_layout_path: Path,
    layout: dict[str, Any],
    tensor: dict[str, Any],
    expected_dim: int,
) -> list[float]:
    if expected_dim <= 0:
        raise PrefillExecuteError("expected vector dim must be positive")
    shape = tensor.get("shape")
    if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int:
        raise PrefillExecuteError("resident vector must have shape [dim]")
    if int(shape[0]) != expected_dim:
        raise PrefillExecuteError(
            f"resident vector dim {shape[0]} does not match expected {expected_dim}"
        )
    dtype = str(tensor.get("dtype") or "")
    dtype_bytes = _dtype_bytes(dtype)
    if dtype_bytes <= 0:
        raise PrefillExecuteError(f"unsupported resident vector dtype {dtype}")
    size = _int_field(tensor, "size", "resident vector")
    if size != expected_dim * dtype_bytes:
        raise PrefillExecuteError(
            f"resident vector size {size} does not match expected "
            f"{expected_dim * dtype_bytes}"
        )
    weight_path = _resident_weight_path_from_layout(resident_layout_path, layout)
    _check_tensor_backing_span(weight_path, tensor, "resident vector")
    offset = _int_field(tensor, "offset", "resident vector")
    try:
        with weight_path.open("rb") as fh:
            fh.seek(offset)
            raw = fh.read(size)
    except OSError as exc:
        raise PrefillExecuteError(
            f"failed to read resident vector from {weight_path}: {exc}"
        ) from exc
    if len(raw) != size:
        raise PrefillExecuteError("short read for resident vector")
    if dtype in {"F32", "float32"}:
        return list(struct.unpack(f"<{expected_dim}f", raw))
    if dtype in {"F16", "float16"}:
        return [float(struct.unpack_from("<e", raw, idx * 2)[0]) for idx in range(expected_dim)]
    if dtype in {"BF16", "bfloat16"}:
        values = []
        for idx in range(expected_dim):
            bits = struct.unpack_from("<H", raw, idx * 2)[0]
            values.append(struct.unpack("<f", (int(bits) << 16).to_bytes(4, "little"))[0])
        return values
    raise PrefillExecuteError(f"unsupported resident vector dtype {dtype}")


def _router_effective_options(
    *,
    layout: dict[str, Any],
    routed_scaling_factor: float | None,
    norm_topk_prob: bool,
    no_norm_topk_prob: bool,
    router_n_group: int | None,
    router_topk_group: int | None,
) -> tuple[float, bool, int, int]:
    meta = layout.get("router")
    if not isinstance(meta, dict):
        meta = {}
    if routed_scaling_factor is None:
        raw_scale = meta.get("routed_scaling_factor")
        scale = float(raw_scale) if isinstance(raw_scale, (int, float)) else 1.0
    else:
        scale = float(routed_scaling_factor)
    if scale <= 0.0 or not math.isfinite(scale):
        raise PrefillExecuteError("routed scaling factor must be positive and finite")
    if norm_topk_prob:
        norm = True
    elif no_norm_topk_prob:
        norm = False
    else:
        raw_norm = meta.get("norm_topk_prob")
        norm = bool(raw_norm) if isinstance(raw_norm, bool) else True
    if router_n_group is None:
        raw_n_group = meta.get("n_group")
        n_group = int(raw_n_group) if type(raw_n_group) is int else 1
    else:
        n_group = int(router_n_group)
    if router_topk_group is None:
        raw_topk_group = meta.get("topk_group")
        topk_group = int(raw_topk_group) if type(raw_topk_group) is int else n_group
    else:
        topk_group = int(router_topk_group)
    if n_group <= 0 or topk_group <= 0 or topk_group > n_group:
        raise PrefillExecuteError("invalid router grouping")
    return scale, norm, n_group, topk_group


def _router_scores(logits: list[float], mode: str) -> list[float]:
    if mode == "softmax":
        maxv = max(logits)
        return [math.exp(value - maxv) for value in logits]
    if mode == "raw":
        return list(logits)
    if mode == "sigmoid":
        return [1.0 / (1.0 + math.exp(-value)) for value in logits]
    raise PrefillExecuteError("router_score must be sigmoid, softmax, or raw")


def _router_topk_with_diagnostics_from_logits(
    logits: list[float],
    *,
    top_k: int,
    router_score: str,
    selection_bias: list[float] | None,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    router_n_group: int,
    router_topk_group: int,
) -> tuple[list[int], list[float], dict[str, Any]]:
    n = len(logits)
    if n <= 0 or top_k <= 0 or top_k > n:
        raise PrefillExecuteError("router top_k must satisfy 1 <= top_k <= experts")
    scores = _router_scores(logits, router_score)
    choice = [
        score + (selection_bias[idx] if selection_bias is not None else 0.0)
        for idx, score in enumerate(scores)
    ]
    allowed = [True] * n
    group_scores: list[float] | None = None
    selected_groups: list[int] | None = None
    group_score_margin: float | None = None
    if router_n_group > 1:
        if n % router_n_group != 0:
            raise PrefillExecuteError("invalid router grouping")
        per_group = n // router_n_group
        group_scores = []
        for group in range(router_n_group):
            values = choice[group * per_group : (group + 1) * per_group]
            best = sorted(values, reverse=True)[:2]
            group_scores.append(sum(best))
        chosen_groups: set[int] = set()
        for _ in range(router_topk_group):
            best_group = 0
            best_score = -math.inf
            for group, score in enumerate(group_scores):
                if group not in chosen_groups and score > best_score:
                    best_group = group
                    best_score = score
            chosen_groups.add(best_group)
        selected_groups = sorted(chosen_groups)
        unselected_group_scores = [
            score for group, score in enumerate(group_scores) if group not in chosen_groups
        ]
        selected_group_scores = [
            score for group, score in enumerate(group_scores) if group in chosen_groups
        ]
        if unselected_group_scores and selected_group_scores:
            group_score_margin = min(selected_group_scores) - max(unselected_group_scores)
        allowed = [False] * n
        for group in chosen_groups:
            for idx in range(group * per_group, (group + 1) * per_group):
                allowed[idx] = True
    used = [False] * n
    indices: list[int] = []
    weights: list[float] = []
    for _ in range(top_k):
        best_idx = 0
        best_score = -math.inf
        for idx, score in enumerate(choice):
            if allowed[idx] and not used[idx] and score > best_score:
                best_idx = idx
                best_score = score
        used[best_idx] = True
        indices.append(best_idx)
        weights.append(scores[best_idx])
    selected_choice_scores = [choice[idx] for idx in indices]
    unselected_allowed_scores = [
        score for idx, score in enumerate(choice) if allowed[idx] and not used[idx]
    ]
    topk_score_margin = (
        min(selected_choice_scores) - max(unselected_allowed_scores)
        if selected_choice_scores and unselected_allowed_scores
        else None
    )
    effective_candidates = [
        value
        for value in (topk_score_margin, group_score_margin)
        if value is not None and math.isfinite(value)
    ]
    diagnostics: dict[str, Any] = {
        "topk_score_margin": topk_score_margin,
        "group_score_margin": group_score_margin,
        "effective_score_margin": min(effective_candidates)
        if effective_candidates
        else None,
        "selected_choice_scores": selected_choice_scores,
    }
    if group_scores is not None:
        diagnostics["group_scores"] = group_scores
    if selected_groups is not None:
        diagnostics["selected_groups"] = selected_groups
    total = sum(weights)
    if norm_topk_prob and total != 0.0:
        weights = [value / total for value in weights]
    weights = [value * routed_scaling_factor for value in weights]
    return indices, weights, diagnostics


def _router_topk_from_logits(
    logits: list[float],
    *,
    top_k: int,
    router_score: str,
    selection_bias: list[float] | None,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    router_n_group: int,
    router_topk_group: int,
) -> tuple[list[int], list[float]]:
    indices, weights, _diagnostics = _router_topk_with_diagnostics_from_logits(
        logits,
        top_k=top_k,
        router_score=router_score,
        selection_bias=selection_bias,
        routed_scaling_factor=routed_scaling_factor,
        norm_topk_prob=norm_topk_prob,
        router_n_group=router_n_group,
        router_topk_group=router_topk_group,
    )
    return indices, weights


def _router_margin_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    def finite_values(key: str) -> list[float]:
        values: list[float] = []
        for record in records:
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            number = float(value)
            if math.isfinite(number):
                values.append(number)
        return values

    summary: dict[str, Any] = {
        "token_count": len(records),
    }
    for source_key, target_prefix in (
        ("topk_score_margin", "topk"),
        ("group_score_margin", "group"),
        ("effective_score_margin", "effective"),
    ):
        values = finite_values(source_key)
        if not values:
            continue
        summary[f"min_{target_prefix}_score_margin"] = min(values)
        summary[f"mean_{target_prefix}_score_margin"] = sum(values) / len(values)
        near_tie_thresholds: dict[str, int] = {}
        for threshold in (1e-6, 1e-5, 1e-4, 1e-3):
            near_tie_thresholds[f"le_{threshold:.0e}"] = sum(
                1 for value in values if value <= threshold
            )
        summary[f"{target_prefix}_near_tie_counts"] = near_tie_thresholds
    if records:
        weakest = min(
            records,
            key=lambda item: (
                float(item.get("effective_score_margin"))
                if isinstance(item.get("effective_score_margin"), (int, float))
                and not isinstance(item.get("effective_score_margin"), bool)
                and math.isfinite(float(item.get("effective_score_margin")))
                else math.inf
            ),
        )
        summary["weakest_token"] = dict(weakest)
    return summary


def _write_router_json_batch_from_logits(
    *,
    resident_layout_path: Path,
    layout: dict[str, Any],
    layer: int,
    logits_path: Path,
    router_json_dir: Path,
    batch_tokens: int,
    num_experts: int,
    top_k: int,
    router_score: str,
    routed_scaling_factor: float | None,
    norm_topk_prob: bool,
    no_norm_topk_prob: bool,
    router_n_group: int | None,
    router_topk_group: int | None,
    ignore_router_bias: bool,
    max_router_mib: int,
) -> dict[str, Any]:
    logits_bytes = batch_tokens * num_experts * 4
    max_router_bytes = int(max_router_mib) * 1024 * 1024
    if logits_bytes > max_router_bytes:
        raise PrefillExecuteError(
            f"router logits bytes {logits_bytes} exceed max_router_mib {max_router_mib}"
        )
    try:
        raw = logits_path.read_bytes()
    except OSError as exc:
        raise PrefillExecuteError(f"failed to read router logits {logits_path}: {exc}") from exc
    if len(raw) != logits_bytes:
        raise PrefillExecuteError(
            f"router logits bytes {len(raw)} do not match expected {logits_bytes}"
        )
    scale, norm, n_group, topk_group = _router_effective_options(
        layout=layout,
        routed_scaling_factor=routed_scaling_factor,
        norm_topk_prob=norm_topk_prob,
        no_norm_topk_prob=no_norm_topk_prob,
        router_n_group=router_n_group,
        router_topk_group=router_topk_group,
    )
    bias_tensor = (
        None
        if ignore_router_bias
        else _find_layer_tensor_optional(
            layout,
            layer=layer,
            suffix=".mlp.gate.e_score_correction_bias",
        )
    )
    bias = (
        _read_resident_vector_f32(
            resident_layout_path=resident_layout_path,
            layout=layout,
            tensor=bias_tensor,
            expected_dim=num_experts,
        )
        if bias_tensor is not None
        else None
    )
    router_json_dir.mkdir(parents=True, exist_ok=True)
    values = struct.unpack(f"<{batch_tokens * num_experts}f", raw)
    margin_records: list[dict[str, Any]] = []
    for token in range(batch_tokens):
        row = list(values[token * num_experts : (token + 1) * num_experts])
        experts, weights, router_margin = _router_topk_with_diagnostics_from_logits(
            row,
            top_k=top_k,
            router_score=router_score,
            selection_bias=bias,
            routed_scaling_factor=scale,
            norm_topk_prob=norm,
            router_n_group=n_group,
            router_topk_group=topk_group,
        )
        payload = {
            "router_score": router_score,
            "top_k": top_k,
            "norm_topk_prob": norm,
            "routed_scaling_factor": scale,
            "n_group": n_group,
            "topk_group": topk_group,
            "used_correction_bias": bias is not None,
            "experts": experts,
            "weights": weights,
            "router_margin": router_margin,
            "logits": row,
        }
        margin_records.append(
            {
                "token_index": token,
                "topk_score_margin": router_margin.get("topk_score_margin"),
                "group_score_margin": router_margin.get("group_score_margin"),
                "effective_score_margin": router_margin.get("effective_score_margin"),
            }
        )
        (router_json_dir / f"token_{token:06d}.router.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
    return _router_margin_summary(margin_records)


def _router_hybrid_margin_threshold_from_env() -> float | None:
    raw = os.environ.get(PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD_ENV)
    if raw is None or raw.strip() == "":
        return None
    try:
        threshold = float(raw)
    except ValueError as exc:
        raise PrefillExecuteError(
            f"{PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD_ENV} must be a positive number"
        ) from exc
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise PrefillExecuteError(
            f"{PREFILL_ROUTER_HYBRID_MARGIN_THRESHOLD_ENV} must be a positive finite number"
        )
    return threshold


def _router_hybrid_margin_threshold(
    value: float | None,
    *,
    label: str = "router_hybrid_margin_threshold",
) -> float | None:
    if value is None:
        return _router_hybrid_margin_threshold_from_env()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefillExecuteError(f"{label} must be a non-negative finite number or None")
    threshold = float(value)
    if threshold == 0.0:
        return _router_hybrid_margin_threshold_from_env()
    if not math.isfinite(threshold) or threshold < 0.0:
        raise PrefillExecuteError(f"{label} must be non-negative and finite")
    return threshold


def _router_summary_min_effective(summary: dict[str, Any]) -> float:
    value = summary.get("min_effective_score_margin")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefillExecuteError("router margin summary missing min_effective_score_margin")
    result = float(value)
    if not math.isfinite(result):
        raise PrefillExecuteError("router min_effective_score_margin must be finite")
    return result


def _with_router_hybrid_policy(
    summary: dict[str, Any],
    *,
    threshold: float,
    decision: str,
    custom_summary: dict[str, Any],
    fallback_summary: dict[str, Any] | None = None,
    custom_elapsed_seconds: float | None = None,
    fallback_elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    result = dict(summary)
    result["router_gate_policy"] = {
        "mode": "custom-first-mpsgraph-fallback",
        "margin_threshold": threshold,
        "decision": decision,
        "command_count": 2 if fallback_elapsed_seconds is not None else 1,
        "custom_elapsed_seconds": custom_elapsed_seconds,
        "fallback_elapsed_seconds": fallback_elapsed_seconds,
        "custom_min_effective_score_margin": _router_summary_min_effective(
            custom_summary
        ),
        "fallback_min_effective_score_margin": (
            _router_summary_min_effective(fallback_summary)
            if fallback_summary is not None
            else None
        ),
    }
    return result


def _run_router_json_batch(
    *,
    runner_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_dir: Path,
    router_json_dir: Path,
    batch_tokens: int,
    hidden_dim: int,
    top_k: int,
    router_score: str,
    routed_scaling_factor: float | None,
    norm_topk_prob: bool,
    no_norm_topk_prob: bool,
    router_n_group: int | None,
    router_topk_group: int | None,
    ignore_router_bias: bool,
    max_resident_matrix_mib: int,
    max_router_mib: int,
    max_runner_scratch_mib: int,
    prefill_linear_backend: str,
    prefill_mpsgraph_min_batch_tokens: int,
    prefill_mpsgraph_min_matrix_dim: int,
    router_hybrid_margin_threshold: float | None,
    keep_token_files: bool,
    echo_runner_output: bool,
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None,
) -> tuple[int, tuple[str, ...], ResidentBatchLinearResult | None, dict[str, Any] | None]:
    runner = Path(runner_path)
    resident_layout = Path(resident_layout_path)
    input_path = Path(input_f32_path)
    token_bytes = hidden_dim * 4
    expected_input_bytes = batch_tokens * token_bytes
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat router input {input_path}: {exc}") from exc
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"router input bytes {input_bytes} do not match expected {expected_input_bytes}"
        )

    router_json_dir.mkdir(parents=True, exist_ok=True)
    layout = _load_json(resident_layout)
    router_matrix = _find_layer_matrix(
        layout,
        layer=layer,
        suffix=PREFILL_ROUTER_GATE_TENSOR_SUFFIX,
    )
    router_rows, router_cols = _tensor_shape2(router_matrix, "router gate")
    if router_cols != hidden_dim:
        raise PrefillExecuteError(
            f"router gate input dim {router_cols} does not match hidden dim {hidden_dim}"
        )
    router_dtype = str(router_matrix.get("dtype") or "")
    resolved_router_backend = _resolve_prefill_linear_backend(
        prefill_linear_backend,
        router_dtype,
        batch_tokens=batch_tokens,
        in_dim=router_cols,
        out_dim=router_rows,
        mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
    )
    hybrid_threshold = _router_hybrid_margin_threshold(
        router_hybrid_margin_threshold
    )
    if (
        hybrid_threshold is not None
        and prefill_linear_backend == "auto"
        and resolved_router_backend == "mpsgraph-f32"
    ):
        custom_logits_path = output_dir / "router_logits.custom.f32"
        custom_gate = run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=resident_layout,
            layer=layer,
            tensor_suffix=PREFILL_ROUTER_GATE_TENSOR_SUFFIX,
            input_f32_path=input_path,
            output_f32_path=custom_logits_path,
            batch_tokens=batch_tokens,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            prefill_linear_backend="custom-metal",
            prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
            prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
            echo_runner_output=echo_runner_output,
            resident_linear_server_session=resident_linear_server_session,
        )
        custom_summary = _write_router_json_batch_from_logits(
            resident_layout_path=resident_layout,
            layout=layout,
            layer=layer,
            logits_path=custom_gate.output_path,
            router_json_dir=router_json_dir,
            batch_tokens=batch_tokens,
            num_experts=router_rows,
            top_k=top_k,
            router_score=router_score,
            routed_scaling_factor=routed_scaling_factor,
            norm_topk_prob=norm_topk_prob,
            no_norm_topk_prob=no_norm_topk_prob,
            router_n_group=router_n_group,
            router_topk_group=router_topk_group,
            ignore_router_bias=ignore_router_bias,
            max_router_mib=max_router_mib,
        )
        custom_min_margin = _router_summary_min_effective(custom_summary)
        if custom_min_margin > hybrid_threshold:
            for token_index in range(batch_tokens):
                router_json = router_json_dir / f"token_{token_index:06d}.router.json"
                if not router_json.exists():
                    raise PrefillExecuteError(f"router JSON was not written: {router_json}")
            return (
                1,
                tuple(custom_gate.command),
                custom_gate,
                _with_router_hybrid_policy(
                    custom_summary,
                    threshold=hybrid_threshold,
                    decision="custom-metal",
                    custom_summary=custom_summary,
                    custom_elapsed_seconds=custom_gate.elapsed_seconds,
                ),
            )

        router_logits_path = output_dir / "router_logits.fallback.mpsgraph.f32"
        fallback_gate = run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=resident_layout,
            layer=layer,
            tensor_suffix=PREFILL_ROUTER_GATE_TENSOR_SUFFIX,
            input_f32_path=input_path,
            output_f32_path=router_logits_path,
            batch_tokens=batch_tokens,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            prefill_linear_backend="mpsgraph-f32",
            prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
            prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
            echo_runner_output=echo_runner_output,
            resident_linear_server_session=resident_linear_server_session,
        )
        fallback_summary = _write_router_json_batch_from_logits(
            resident_layout_path=resident_layout,
            layout=layout,
            layer=layer,
            logits_path=fallback_gate.output_path,
            router_json_dir=router_json_dir,
            batch_tokens=batch_tokens,
            num_experts=router_rows,
            top_k=top_k,
            router_score=router_score,
            routed_scaling_factor=routed_scaling_factor,
            norm_topk_prob=norm_topk_prob,
            no_norm_topk_prob=no_norm_topk_prob,
            router_n_group=router_n_group,
            router_topk_group=router_topk_group,
            ignore_router_bias=ignore_router_bias,
            max_router_mib=max_router_mib,
        )
        for token_index in range(batch_tokens):
            router_json = router_json_dir / f"token_{token_index:06d}.router.json"
            if not router_json.exists():
                raise PrefillExecuteError(f"router JSON was not written: {router_json}")
        return (
            2,
            tuple(custom_gate.command),
            fallback_gate,
            _with_router_hybrid_policy(
                fallback_summary,
                threshold=hybrid_threshold,
                decision="mpsgraph-f32-fallback",
                custom_summary=custom_summary,
                fallback_summary=fallback_summary,
                custom_elapsed_seconds=custom_gate.elapsed_seconds,
                fallback_elapsed_seconds=fallback_gate.elapsed_seconds,
            ),
        )

    if resolved_router_backend in PREFILL_LINEAR_ACCELERATED_BACKENDS:
        router_logits_path = output_dir / "router_logits.f32"
        router_gate = run_resident_batch_linear(
            runner_path=runner,
            resident_layout_path=resident_layout,
            layer=layer,
            tensor_suffix=PREFILL_ROUTER_GATE_TENSOR_SUFFIX,
            input_f32_path=input_path,
            output_f32_path=router_logits_path,
            batch_tokens=batch_tokens,
            max_resident_matrix_mib=max_resident_matrix_mib,
            max_runner_scratch_mib=max_runner_scratch_mib,
            prefill_linear_backend=prefill_linear_backend,
            prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
            prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
            echo_runner_output=echo_runner_output,
            resident_linear_server_session=resident_linear_server_session,
        )
        router_margin_summary = _write_router_json_batch_from_logits(
            resident_layout_path=resident_layout,
            layout=layout,
            layer=layer,
            logits_path=router_gate.output_path,
            router_json_dir=router_json_dir,
            batch_tokens=batch_tokens,
            num_experts=router_rows,
            top_k=top_k,
            router_score=router_score,
            routed_scaling_factor=routed_scaling_factor,
            norm_topk_prob=norm_topk_prob,
            no_norm_topk_prob=no_norm_topk_prob,
            router_n_group=router_n_group,
            router_topk_group=router_topk_group,
            ignore_router_bias=ignore_router_bias,
            max_router_mib=max_router_mib,
        )
        for token_index in range(batch_tokens):
            router_json = router_json_dir / f"token_{token_index:06d}.router.json"
            if not router_json.exists():
                raise PrefillExecuteError(f"router JSON was not written: {router_json}")
        return 1, tuple(router_gate.command), router_gate, router_margin_summary

    cmd = [
        str(runner),
        "--resident-layout",
        str(resident_layout),
        "--layer",
        str(layer),
        "--run-router-batch",
        "--input-f32",
        str(input_path),
        "--batch-tokens",
        str(batch_tokens),
        "--top-k",
        str(top_k),
        "--router-score",
        router_score,
        "--output-router-json-dir",
        str(router_json_dir),
        "--max-router-mib",
        str(max_router_mib),
        "--max-runner-scratch-mib",
        str(max_runner_scratch_mib),
    ]
    if routed_scaling_factor is not None:
        cmd.extend(["--routed-scaling-factor", f"{routed_scaling_factor:.9g}"])
    if norm_topk_prob:
        cmd.append("--norm-topk-prob")
    if no_norm_topk_prob:
        cmd.append("--no-norm-topk-prob")
    if router_n_group is not None:
        cmd.extend(["--router-n-group", str(router_n_group)])
    if router_topk_group is not None:
        cmd.extend(["--router-topk-group", str(router_topk_group)])
    if ignore_router_bias:
        cmd.append("--ignore-router-bias")
    _run_command(cmd, echo_output=echo_runner_output)
    for token_index in range(batch_tokens):
        router_json = router_json_dir / f"token_{token_index:06d}.router.json"
        if not router_json.exists():
            raise PrefillExecuteError(f"router JSON was not written: {router_json}")
    _ = output_dir
    _ = keep_token_files
    return 1, tuple(cmd), None, None


def run_prefill_staged_routed_mlp_block_batch(
    *,
    runner_path: str | Path,
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_dir: str | Path,
    output_f32_path: str | Path,
    batch_tokens: int,
    top_k: int = 8,
    max_k: int = 8,
    router_score: str = "sigmoid",
    routed_scaling_factor: float | None = None,
    norm_topk_prob: bool = False,
    no_norm_topk_prob: bool = False,
    router_n_group: int | None = None,
    router_topk_group: int | None = None,
    ignore_router_bias: bool = False,
    include_shared_expert: bool = False,
    rms_norm_eps: float = 1e-5,
    max_resident_matrix_mib: int = 512,
    max_slot_mib: int = 256,
    max_router_mib: int = 64,
    max_runner_scratch_mib: int = 4096,
    expert_stage_merge_gap_kib: float = 0.0,
    expert_stage_align_kib: float = 4.0,
    max_stage_mib: float = 4096.0,
    max_compact_stage_mib: float = 4096.0,
    copy_chunk_mib: float = 8.0,
    stage_disk_safety_margin_bytes: int = 0,
    prefill_ssd_read_gib_per_second: float = 0.0,
    prefill_max_routed_read_seconds: float = 0.0,
    expert_stage_max_raw_ranges: int = 0,
    expert_stage_max_coalesced_ranges: int = 0,
    expert_stage_tiling: bool = False,
    moe_token_block: MoETokenBlock = "auto",
    moe_output_accumulator: MoEOutputAccumulator = "env",
    static_capacity_per_expert: int | None = None,
    static_capacity_output_json_path: str | Path | None = None,
    static_capacity_output_bin_path: str | Path | None = None,
    write_static_capacity_json: bool = True,
    allow_static_capacity_overflow: bool = False,
    shared_expert_server_session: ResidentSharedExpertBatchServerSession | None = None,
    prefill_linear_backend: str = "custom-metal",
    prefill_mpsgraph_min_batch_tokens: int = AUTO_MPSGRAPH_MIN_BATCH_TOKENS,
    prefill_mpsgraph_min_matrix_dim: int = AUTO_MPSGRAPH_MIN_DIM,
    router_hybrid_margin_threshold: float | None = None,
    keep_token_files: bool = False,
    echo_runner_output: bool = True,
    moe_plan_server_session: StagedRoutedMoEBatchPlanServerSession | None = None,
    resident_linear_server_session: ResidentBatchLinearServerSession | None = None,
    rmsnorm_server_session: ResidentBatchRMSNormServerSession | None = None,
) -> PrefillStagedRoutedMLPBlockBatchResult:
    layer = _require_nonnegative_int(layer, "layer")
    batch_tokens = _require_positive_int(batch_tokens, "batch_tokens")
    top_k = _require_positive_int(top_k, "top_k")
    max_k = _require_positive_int(max_k, "max_k")
    if top_k <= 0 or max_k <= 0 or top_k > max_k or max_k > 64:
        raise PrefillExecuteError("top_k/max_k must satisfy 1 <= top_k <= max_k <= 64")
    if router_score not in {"sigmoid", "softmax", "raw"}:
        raise PrefillExecuteError("router_score must be sigmoid, softmax, or raw")
    if norm_topk_prob and no_norm_topk_prob:
        raise PrefillExecuteError("norm_topk_prob and no_norm_topk_prob conflict")
    routed_scaling_factor = _optional_positive_number(
        routed_scaling_factor,
        "routed_scaling_factor",
    )
    router_n_group = _optional_positive_int(router_n_group, "router_n_group")
    router_topk_group = _optional_positive_int(
        router_topk_group,
        "router_topk_group",
    )
    rms_norm_eps = _require_number(rms_norm_eps, "rms_norm_eps")
    if rms_norm_eps < 0:
        raise PrefillExecuteError("rms_norm_eps must be non-negative")
    max_resident_matrix_mib = _positive_integer_mib(
        "max_resident_matrix_mib",
        max_resident_matrix_mib,
    )
    max_slot_mib = _positive_integer_mib("max_slot_mib", max_slot_mib)
    max_router_mib = _positive_integer_mib("max_router_mib", max_router_mib)
    max_runner_scratch_mib = _positive_integer_mib(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )
    expert_stage_merge_gap_kib = _require_nonnegative_number(
        expert_stage_merge_gap_kib,
        "expert_stage_merge_gap_kib",
    )
    expert_stage_align_kib = _require_positive_number(
        expert_stage_align_kib,
        "expert_stage_align_kib",
    )
    max_stage_mib = _require_positive_number(max_stage_mib, "max_stage_mib")
    max_compact_stage_mib = _require_positive_number(
        max_compact_stage_mib,
        "max_compact_stage_mib",
    )
    copy_chunk_mib = _require_positive_number(copy_chunk_mib, "copy_chunk_mib")
    stage_disk_safety_margin_bytes = _require_nonnegative_int(
        stage_disk_safety_margin_bytes,
        "stage_disk_safety_margin_bytes",
    )
    prefill_ssd_read_gib_per_second = _require_nonnegative_number(
        prefill_ssd_read_gib_per_second,
        "prefill_ssd_read_gib_per_second",
    )
    prefill_max_routed_read_seconds = _require_nonnegative_number(
        prefill_max_routed_read_seconds,
        "prefill_max_routed_read_seconds",
    )
    expert_stage_max_raw_ranges = _require_nonnegative_int(
        expert_stage_max_raw_ranges,
        "expert_stage_max_raw_ranges",
    )
    expert_stage_max_coalesced_ranges = _require_nonnegative_int(
        expert_stage_max_coalesced_ranges,
        "expert_stage_max_coalesced_ranges",
    )
    if type(expert_stage_tiling) is not bool:
        raise PrefillExecuteError("expert_stage_tiling must be a boolean")
    router_hybrid_margin_threshold = _router_hybrid_margin_threshold(
        router_hybrid_margin_threshold
    )
    if (
        prefill_max_routed_read_seconds > 0
        and prefill_ssd_read_gib_per_second <= 0
    ):
        raise PrefillExecuteError(
            "prefill_ssd_read_gib_per_second must be positive when "
            "prefill_max_routed_read_seconds is set"
        )
    try:
        _normalize_moe_token_block(moe_token_block)
        moe_output_accumulator = _normalize_moe_output_accumulator(
            moe_output_accumulator
        )
    except StagedMoEError as exc:
        raise PrefillExecuteError(str(exc)) from exc
    static_capacity_per_expert = _optional_positive_int(
        static_capacity_per_expert,
        "static_capacity_per_expert",
    )
    stage_merge_gap_bytes = int(expert_stage_merge_gap_kib * 1024)
    stage_align_bytes = int(expert_stage_align_kib * 1024)
    if stage_align_bytes <= 0:
        raise PrefillExecuteError("expert stage alignment must be at least one byte")

    max_slot_bytes = max_slot_mib * 1024 * 1024
    max_router_bytes = max_router_mib * 1024 * 1024
    max_resident_matrix_bytes = max_resident_matrix_mib * 1024 * 1024
    max_runner_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    try:
        budget = check_layer_runtime(
            expert_layout_path=expert_layout_path,
            resident_layout_path=resident_layout_path,
            layer=layer,
            top_k=top_k,
            max_k=max_k,
            max_slot_bytes=max_slot_bytes,
            max_router_bytes=max_router_bytes,
            max_resident_matrix_bytes=max_resident_matrix_bytes,
            max_runner_scratch_bytes=max_runner_scratch_bytes,
            include_shared_expert=include_shared_expert,
        )
    except RuntimeCheckError as exc:
        raise PrefillExecuteError(f"staged routed MLP preflight failed: {exc}") from exc

    input_path = Path(input_f32_path)
    expected_input_bytes = batch_tokens * budget.hidden_dim * 4
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat staged routed MLP input {input_path}: {exc}") from exc
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"input bytes {input_bytes} do not match batch_tokens*hidden_dim*f32 "
            f"({expected_input_bytes})"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_output = out_dir / "post_attention_layernorm.f32"
    router_json_dir = out_dir / "router_json"
    stage_file = out_dir / "experts.stage.bin"
    stage_manifest = out_dir / "experts.stage.manifest.json"
    staged_output_dir = out_dir / "staged_moe"
    routed_output = out_dir / "routed_moe.f32"
    shared_gate_output = out_dir / "shared_gate_proj.f32"
    shared_up_output = out_dir / "shared_up_proj.f32"
    shared_swiglu_output = out_dir / "shared_swiglu.f32"
    shared_output = out_dir / "shared_down_proj.f32"
    routed_shared_output = out_dir / "routed_plus_shared.f32"

    norm = run_resident_batch_rmsnorm(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        norm_suffix=".post_attention_layernorm.weight",
        input_f32_path=input_path,
        output_f32_path=norm_output,
        batch_tokens=batch_tokens,
        rms_norm_eps=rms_norm_eps,
        max_runner_scratch_mib=max_runner_scratch_mib,
        echo_runner_output=echo_runner_output,
        rmsnorm_server_session=rmsnorm_server_session,
    )
    (
        router_command_count,
        first_router_command,
        router_gate,
        router_margin_summary,
    ) = _run_router_json_batch(
        runner_path=runner_path,
        resident_layout_path=resident_layout_path,
        layer=layer,
        input_f32_path=norm.output_path,
        output_dir=out_dir,
        router_json_dir=router_json_dir,
        batch_tokens=batch_tokens,
        hidden_dim=norm.hidden_dim,
        top_k=top_k,
        router_score=router_score,
        routed_scaling_factor=routed_scaling_factor,
        norm_topk_prob=norm_topk_prob,
        no_norm_topk_prob=no_norm_topk_prob,
        router_n_group=router_n_group,
        router_topk_group=router_topk_group,
        ignore_router_bias=ignore_router_bias,
        max_resident_matrix_mib=max_resident_matrix_mib,
        max_router_mib=max_router_mib,
        max_runner_scratch_mib=max_runner_scratch_mib,
        prefill_linear_backend=prefill_linear_backend,
        prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
        prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
        router_hybrid_margin_threshold=router_hybrid_margin_threshold,
        keep_token_files=keep_token_files,
        echo_runner_output=echo_runner_output,
        resident_linear_server_session=resident_linear_server_session,
    )
    tiled_staged_moe: TiledStagedRoutedMoEBatchResult | None = None
    try:
        if expert_stage_tiling:
            if (
                static_capacity_output_json_path is not None
                or static_capacity_output_bin_path is not None
            ):
                raise PrefillExecuteError(
                    "static capacity output paths are not supported with "
                    "expert_stage_tiling because each tile writes its own route table"
                )
            routed_moe_started = time.perf_counter()
            tiled_staged_moe = run_tiled_staged_routed_moe_batch(
                runner_path=runner_path,
                expert_layout_path=expert_layout_path,
                layer=layer,
                router_json_dir=router_json_dir,
                input_f32_path=norm.output_path,
                output_f32_path=routed_output,
                output_dir=staged_output_dir,
                merge_gap_bytes=stage_merge_gap_bytes,
                align_bytes=stage_align_bytes,
                max_stage_mib=max_stage_mib,
                max_compact_stage_mib=max_compact_stage_mib,
                copy_chunk_mib=copy_chunk_mib,
                disk_safety_margin_bytes=stage_disk_safety_margin_bytes,
                ssd_read_gib_per_second=prefill_ssd_read_gib_per_second,
                max_read_seconds=prefill_max_routed_read_seconds,
                max_raw_ranges=expert_stage_max_raw_ranges,
                max_coalesced_ranges=expert_stage_max_coalesced_ranges,
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
            routed_moe_elapsed = time.perf_counter() - routed_moe_started
            if (
                not tiled_staged_moe.tile_stage_results
                or not tiled_staged_moe.tile_results
            ):
                raise PrefillExecuteError("tiled staged MoE produced no tiles")
            stage_result = tiled_staged_moe.tile_stage_results[0]
            staged_moe = tiled_staged_moe.tile_results[0]
        else:
            stage_result = stage_batch_experts(
                expert_layout_path,
                layer=layer,
                router_json_dir=router_json_dir,
                stage_file_path=stage_file,
                manifest_path=stage_manifest,
                merge_gap_bytes=stage_merge_gap_bytes,
                align_bytes=stage_align_bytes,
                max_stage_mib=max_stage_mib,
                copy_chunk_mib=copy_chunk_mib,
                disk_safety_margin_bytes=stage_disk_safety_margin_bytes,
                ssd_read_gib_per_second=prefill_ssd_read_gib_per_second,
                max_read_seconds=prefill_max_routed_read_seconds,
                max_raw_ranges=expert_stage_max_raw_ranges,
                max_coalesced_ranges=expert_stage_max_coalesced_ranges,
            )
            routed_moe_started = time.perf_counter()
            staged_moe = run_staged_routed_moe_batch(
                runner_path=runner_path,
                stage_manifest_path=stage_manifest,
                input_f32_path=norm.output_path,
                output_f32_path=routed_output,
                output_dir=staged_output_dir,
                max_compact_stage_mib=max_compact_stage_mib,
                copy_chunk_mib=copy_chunk_mib,
                disk_safety_margin_bytes=stage_disk_safety_margin_bytes,
                max_slot_mib=max_slot_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                moe_token_block=moe_token_block,
                static_capacity_per_expert=static_capacity_per_expert,
                static_capacity_output_json_path=static_capacity_output_json_path,
                static_capacity_output_bin_path=static_capacity_output_bin_path,
                write_static_capacity_json=write_static_capacity_json,
                allow_static_capacity_overflow=allow_static_capacity_overflow,
                keep_token_files=keep_token_files,
                echo_runner_output=echo_runner_output,
                moe_plan_server_session=moe_plan_server_session,
                moe_output_accumulator=moe_output_accumulator,
            )
            routed_moe_elapsed = time.perf_counter() - routed_moe_started
    except (ExpertIOPlanError, StagedMoEError) as exc:
        raise PrefillExecuteError(f"staged routed MoE failed: {exc}") from exc

    if tiled_staged_moe is None:
        routed_output_bytes = staged_moe.output_bytes
        staged_bytes = stage_result.staged_bytes
        compact_stage_bytes = staged_moe.compact_stage_bytes
        compact_stage_materialized_bytes = staged_moe.compact_stage_materialized_bytes
        compact_stage_storage = staged_moe.compact_stage_storage
        static_capacity_path = staged_moe.static_capacity_path
        static_capacity_binary_path = staged_moe.static_capacity_binary_path
        static_capacity_per_expert_result = staged_moe.static_capacity_per_expert
        static_capacity_used_slots = staged_moe.static_capacity_used_slots
        static_capacity_total_slots = staged_moe.static_capacity_total_slots
        static_capacity_overflow_assignments = (
            staged_moe.static_capacity_overflow_assignments
        )
        static_capacity_binary_bytes = staged_moe.static_capacity_binary_bytes
        routed_command_count = staged_moe.command_count
        stage_copy_elapsed_seconds = stage_result.copy_elapsed_seconds
        stage_copy_throughput = stage_result.copy_throughput_gib_per_second
        stage_copy_seconds_ok = stage_result.io_summary.copy_seconds_ok
        max_stage_copy_chunk_bytes = stage_result.copy_chunk_bytes
        max_moe_copy_chunk_bytes = staged_moe.copy_chunk_bytes
        max_moe_estimated_peak_bytes = staged_moe.moe_estimated_peak_bytes or 0
    else:
        tile_stage_results = tiled_staged_moe.tile_stage_results
        tile_moe_results = tiled_staged_moe.tile_results
        routed_output_bytes = tiled_staged_moe.output_bytes
        staged_bytes = sum(item.staged_bytes for item in tile_stage_results)
        compact_stage_bytes = sum(item.compact_stage_bytes for item in tile_moe_results)
        compact_stage_materialized_bytes = sum(
            item.compact_stage_materialized_bytes for item in tile_moe_results
        )
        compact_stage_storage = "tiled"
        static_capacity_path = None
        static_capacity_binary_path = None
        static_capacity_values = [
            item.static_capacity_per_expert
            for item in tile_moe_results
            if item.static_capacity_per_expert is not None
        ]
        static_capacity_per_expert_result = (
            max(static_capacity_values) if static_capacity_values else None
        )
        static_capacity_used_slots = sum(
            item.static_capacity_used_slots for item in tile_moe_results
        )
        static_capacity_total_slots = sum(
            item.static_capacity_total_slots for item in tile_moe_results
        )
        static_capacity_overflow_assignments = sum(
            item.static_capacity_overflow_assignments for item in tile_moe_results
        )
        static_capacity_binary_bytes = sum(
            item.static_capacity_binary_bytes for item in tile_moe_results
        )
        routed_command_count = sum(item.command_count for item in tile_moe_results)
        copy_elapsed_values = [
            item.copy_elapsed_seconds
            for item in tile_stage_results
            if item.copy_elapsed_seconds is not None
        ]
        stage_copy_elapsed_seconds = (
            sum(copy_elapsed_values) if copy_elapsed_values else None
        )
        stage_copy_throughput = (
            (staged_bytes / 1024**3) / stage_copy_elapsed_seconds
            if (
                stage_copy_elapsed_seconds is not None
                and stage_copy_elapsed_seconds > 0
            )
            else None
        )
        copy_ok_values = [
            item.io_summary.copy_seconds_ok
            for item in tile_stage_results
            if item.io_summary.copy_seconds_ok is not None
        ]
        stage_copy_seconds_ok = (
            all(item is True for item in copy_ok_values) if copy_ok_values else None
        )
        max_stage_copy_chunk_bytes = max(
            (item.copy_chunk_bytes for item in tile_stage_results),
            default=0,
        )
        max_moe_copy_chunk_bytes = max(
            (item.copy_chunk_bytes for item in tile_moe_results),
            default=0,
        )
        max_moe_estimated_peak_bytes = max(
            (item.moe_estimated_peak_bytes or 0 for item in tile_moe_results),
            default=0,
        )

    shared_gate: ResidentBatchLinearResult | None = None
    shared_up: ResidentBatchLinearResult | None = None
    shared_down: ResidentBatchLinearResult | None = None
    shared_expert_batch: ResidentSharedExpertBatchResult | None = None
    shared_output_path: Path | None = None
    shared_output_bytes = 0
    shared_swiglu_peak = 0
    shared_add_peak = 0
    residual_lhs_path = routed_output
    if include_shared_expert:
        used_fused_shared_expert = False
        if (
            _fused_shared_expert_batch_enabled()
            and prefill_linear_backend in {"custom-metal", "auto"}
            and _shared_expert_fused_candidate_available(
                resident_layout_path,
                layer=layer,
            )
        ):
            shared_expert_batch = run_resident_shared_expert_batch(
                runner_path=runner_path,
                resident_layout_path=resident_layout_path,
                layer=layer,
                input_f32_path=norm.output_path,
                output_f32_path=shared_output,
                batch_tokens=batch_tokens,
                max_resident_matrix_mib=max_resident_matrix_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                shared_expert_server_session=shared_expert_server_session,
                echo_runner_output=echo_runner_output,
            )
            if shared_expert_batch.hidden_dim != norm.hidden_dim:
                raise PrefillExecuteError(
                    "fused shared expert output dim does not match hidden dim"
                )
            shared_swiglu_peak = shared_expert_batch.activation_bytes
            shared_output_bytes, shared_add_peak = _add_f32_batches_streaming(
                lhs_path=routed_output,
                rhs_path=shared_expert_batch.output_path,
                output_path=routed_shared_output,
                batch_tokens=batch_tokens,
                dim=norm.hidden_dim,
            )
            shared_output_path = shared_expert_batch.output_path
            residual_lhs_path = routed_shared_output
            used_fused_shared_expert = True

        if not used_fused_shared_expert:
            shared_gate_suffix, shared_up_suffix, shared_down_suffix = _shared_expert_suffixes(
                resident_layout_path,
                layer=layer,
            )
            shared_gate = run_resident_batch_linear(
                runner_path=runner_path,
                resident_layout_path=resident_layout_path,
                layer=layer,
                tensor_suffix=shared_gate_suffix,
                input_f32_path=norm.output_path,
                output_f32_path=shared_gate_output,
                batch_tokens=batch_tokens,
                max_resident_matrix_mib=max_resident_matrix_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                prefill_linear_backend=prefill_linear_backend,
                prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
                prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
                echo_runner_output=echo_runner_output,
                resident_linear_server_session=resident_linear_server_session,
            )
            shared_up = run_resident_batch_linear(
                runner_path=runner_path,
                resident_layout_path=resident_layout_path,
                layer=layer,
                tensor_suffix=shared_up_suffix,
                input_f32_path=norm.output_path,
                output_f32_path=shared_up_output,
                batch_tokens=batch_tokens,
                max_resident_matrix_mib=max_resident_matrix_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                prefill_linear_backend=prefill_linear_backend,
                prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
                prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
                echo_runner_output=echo_runner_output,
                resident_linear_server_session=resident_linear_server_session,
            )
            if shared_gate.in_dim != norm.hidden_dim or shared_up.in_dim != norm.hidden_dim:
                raise PrefillExecuteError(
                    "shared expert gate/up input dims do not match hidden dim"
                )
            if shared_gate.out_dim != shared_up.out_dim:
                raise PrefillExecuteError(
                    f"shared expert gate dim {shared_gate.out_dim} does not match "
                    f"up dim {shared_up.out_dim}"
                )
            _shared_swiglu_bytes, shared_swiglu_peak = _swiglu_f32_batches_streaming(
                gate_path=shared_gate.output_path,
                up_path=shared_up.output_path,
                output_path=shared_swiglu_output,
                batch_tokens=batch_tokens,
                dim=shared_gate.out_dim,
            )
            shared_down = run_resident_batch_linear(
                runner_path=runner_path,
                resident_layout_path=resident_layout_path,
                layer=layer,
                tensor_suffix=shared_down_suffix,
                input_f32_path=shared_swiglu_output,
                output_f32_path=shared_output,
                batch_tokens=batch_tokens,
                max_resident_matrix_mib=max_resident_matrix_mib,
                max_runner_scratch_mib=max_runner_scratch_mib,
                prefill_linear_backend=prefill_linear_backend,
                prefill_mpsgraph_min_batch_tokens=prefill_mpsgraph_min_batch_tokens,
                prefill_mpsgraph_min_matrix_dim=prefill_mpsgraph_min_matrix_dim,
                echo_runner_output=echo_runner_output,
                resident_linear_server_session=resident_linear_server_session,
            )
            if (
                shared_down.in_dim != shared_gate.out_dim
                or shared_down.out_dim != norm.hidden_dim
            ):
                raise PrefillExecuteError(
                    f"shared expert down shape [{shared_down.out_dim},{shared_down.in_dim}] "
                    f"does not match [{norm.hidden_dim},{shared_gate.out_dim}]"
                )
            shared_output_bytes, shared_add_peak = _add_f32_batches_streaming(
                lhs_path=routed_output,
                rhs_path=shared_down.output_path,
                output_path=routed_shared_output,
                batch_tokens=batch_tokens,
                dim=norm.hidden_dim,
            )
            shared_output_path = shared_down.output_path
            residual_lhs_path = routed_shared_output

    output_bytes, add_peak = _add_f32_batches_streaming(
        lhs_path=residual_lhs_path,
        rhs_path=input_path,
        output_path=Path(output_f32_path),
        batch_tokens=batch_tokens,
        dim=norm.hidden_dim,
    )
    estimated_peak = max(
        budget.estimated_peak_bytes,
        norm.estimated_peak_bytes,
        router_gate.estimated_peak_bytes if router_gate is not None else 0,
        max_stage_copy_chunk_bytes,
        max_moe_copy_chunk_bytes,
        max_moe_estimated_peak_bytes,
        shared_expert_batch.estimated_peak_bytes if shared_expert_batch is not None else 0,
        shared_gate.estimated_peak_bytes if shared_gate is not None else 0,
        shared_up.estimated_peak_bytes if shared_up is not None else 0,
        shared_swiglu_peak,
        shared_down.estimated_peak_bytes if shared_down is not None else 0,
        shared_add_peak,
        add_peak,
    )
    return PrefillStagedRoutedMLPBlockBatchResult(
        runner_path=Path(runner_path),
        expert_layout_path=Path(expert_layout_path),
        resident_layout_path=Path(resident_layout_path),
        input_path=input_path,
        output_dir=out_dir,
        output_path=Path(output_f32_path),
        router_json_dir=router_json_dir,
        stage_file_path=stage_result.stage_file_path,
        stage_manifest_path=stage_result.manifest_path or stage_manifest,
        routed_output_path=routed_output,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=norm.hidden_dim,
        input_bytes=input_bytes,
        norm_output_bytes=norm.output_bytes,
        routed_output_bytes=routed_output_bytes,
        output_bytes=output_bytes,
        top_k=top_k,
        max_k=max_k,
        router_score=router_score,
        include_shared_expert=include_shared_expert,
        rms_norm_eps=rms_norm_eps,
        staged_bytes=staged_bytes,
        compact_stage_bytes=compact_stage_bytes,
        compact_stage_materialized_bytes=compact_stage_materialized_bytes,
        compact_stage_storage=compact_stage_storage,
        stage_plus_compact_bytes=(
            staged_bytes + compact_stage_bytes
        ),
        stage_plus_compact_materialized_bytes=(
            staged_bytes + compact_stage_materialized_bytes
        ),
        static_capacity_path=static_capacity_path,
        static_capacity_binary_path=static_capacity_binary_path,
        static_capacity_per_expert=static_capacity_per_expert_result,
        static_capacity_used_slots=static_capacity_used_slots,
        static_capacity_total_slots=static_capacity_total_slots,
        static_capacity_overflow_assignments=static_capacity_overflow_assignments,
        static_capacity_binary_bytes=static_capacity_binary_bytes,
        shared_output_path=shared_output_path,
        shared_output_bytes=shared_output_bytes,
        shared_swiglu_peak_bytes=shared_swiglu_peak,
        shared_add_peak_bytes=shared_add_peak,
        residual_add_peak_bytes=add_peak,
        estimated_peak_bytes=estimated_peak,
        router_command_count=router_command_count,
        routed_command_count=routed_command_count,
        first_router_command=first_router_command,
        router_margin_summary=router_margin_summary,
        post_attention_layernorm=norm,
        router_gate_proj=router_gate,
        stage_result=stage_result,
        staged_moe=staged_moe,
        tiled_staged_moe=tiled_staged_moe,
        shared_expert_batch=shared_expert_batch,
        shared_gate_proj=shared_gate,
        shared_up_proj=shared_up,
        shared_down_proj=shared_down,
        stage_copy_elapsed_seconds=stage_copy_elapsed_seconds,
        stage_copy_throughput_gib_per_second=stage_copy_throughput,
        stage_copy_seconds_ok=stage_copy_seconds_ok,
        routed_moe_elapsed_seconds=routed_moe_elapsed,
    )


def run_prefill_routed_mlp_block_batch(
    *,
    runner_path: str | Path,
    expert_layout_path: str | Path,
    resident_layout_path: str | Path,
    layer: int,
    input_f32_path: str | Path,
    output_dir: str | Path,
    output_f32_path: str | Path,
    batch_tokens: int,
    top_k: int = 8,
    max_k: int = 8,
    router_score: str = "sigmoid",
    routed_scaling_factor: float | None = None,
    norm_topk_prob: bool = False,
    no_norm_topk_prob: bool = False,
    router_n_group: int | None = None,
    router_topk_group: int | None = None,
    ignore_router_bias: bool = False,
    include_shared_expert: bool = False,
    rms_norm_eps: float = 1e-5,
    max_slot_mib: int = 256,
    max_router_mib: int = 64,
    max_runner_scratch_mib: int = 4096,
    expert_read_advise_merge_gap_kib: int = 0,
    expert_read_advise_align_kib: int = 0,
    router_json_dir: str | Path | None = None,
    keep_token_files: bool = False,
    echo_runner_output: bool = True,
) -> PrefillRoutedMLPBlockBatchResult:
    layer = _require_nonnegative_int(layer, "layer")
    batch_tokens = _require_positive_int(batch_tokens, "batch_tokens")
    top_k = _require_positive_int(top_k, "top_k")
    max_k = _require_positive_int(max_k, "max_k")
    if top_k <= 0 or max_k <= 0 or top_k > max_k or max_k > 64:
        raise PrefillExecuteError("top_k/max_k must satisfy 1 <= top_k <= max_k <= 64")
    if router_score not in {"sigmoid", "softmax", "raw"}:
        raise PrefillExecuteError("router_score must be sigmoid, softmax, or raw")
    if norm_topk_prob and no_norm_topk_prob:
        raise PrefillExecuteError("norm_topk_prob and no_norm_topk_prob conflict")
    routed_scaling_factor = _optional_positive_number(
        routed_scaling_factor,
        "routed_scaling_factor",
    )
    router_n_group = _optional_positive_int(router_n_group, "router_n_group")
    router_topk_group = _optional_positive_int(
        router_topk_group,
        "router_topk_group",
    )
    rms_norm_eps = _require_number(rms_norm_eps, "rms_norm_eps")
    if rms_norm_eps < 0:
        raise PrefillExecuteError("rms_norm_eps must be non-negative")
    max_slot_mib = _positive_integer_mib("max_slot_mib", max_slot_mib)
    max_router_mib = _positive_integer_mib("max_router_mib", max_router_mib)
    max_runner_scratch_mib = _positive_integer_mib(
        "max_runner_scratch_mib",
        max_runner_scratch_mib,
    )
    expert_read_advise_merge_gap_kib = _require_nonnegative_int(
        expert_read_advise_merge_gap_kib,
        "expert_read_advise_merge_gap_kib",
    )
    expert_read_advise_align_kib = _require_nonnegative_int(
        expert_read_advise_align_kib,
        "expert_read_advise_align_kib",
    )

    max_slot_bytes = max_slot_mib * 1024 * 1024
    max_router_bytes = max_router_mib * 1024 * 1024
    max_runner_scratch_bytes = max_runner_scratch_mib * 1024 * 1024
    try:
        budget = check_layer_runtime(
            expert_layout_path=expert_layout_path,
            resident_layout_path=resident_layout_path,
            layer=layer,
            top_k=top_k,
            max_k=max_k,
            max_slot_bytes=max_slot_bytes,
            max_router_bytes=max_router_bytes,
            max_runner_scratch_bytes=max_runner_scratch_bytes,
            include_shared_expert=include_shared_expert,
        )
    except RuntimeCheckError as exc:
        raise PrefillExecuteError(f"routed MLP batch preflight failed: {exc}") from exc

    input_path = Path(input_f32_path)
    token_bytes = budget.hidden_dim * 4
    expected_input_bytes = batch_tokens * token_bytes
    try:
        input_bytes = input_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat routed MLP input {input_path}: {exc}") from exc
    if input_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"input bytes {input_bytes} do not match batch_tokens*hidden_dim*f32 "
            f"({expected_input_bytes})"
        )

    runner = Path(runner_path)
    if not runner.exists():
        raise PrefillExecuteError(f"runner not found: {runner}")
    expert_layout = Path(expert_layout_path)
    resident_layout = Path(resident_layout_path)
    out_dir = Path(output_dir)
    token_dir = out_dir / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_f32_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    router_dir = Path(router_json_dir) if router_json_dir is not None else None
    if router_dir is not None:
        router_dir.mkdir(parents=True, exist_ok=True)

    first_command: tuple[str, ...] = ()
    try:
        with input_path.open("rb") as source, output_path.open("wb") as batch_out:
            for token_index in range(batch_tokens):
                row = source.read(token_bytes)
                if len(row) != token_bytes:
                    raise PrefillExecuteError("failed to read full routed MLP input row")
                token_stem = f"token_{token_index:06d}"
                token_input = token_dir / f"{token_stem}.input.f32"
                token_output = token_dir / f"{token_stem}.output.f32"
                token_input.write_bytes(row)
                cmd = [
                    str(runner),
                    "--layout",
                    str(expert_layout),
                    "--resident-layout",
                    str(resident_layout),
                    "--layer",
                    str(layer),
                    "--run-mlp-block",
                    "--input-f32",
                    str(token_input),
                    "--top-k",
                    str(top_k),
                    "--max-k",
                    str(max_k),
                    "--router-score",
                    router_score,
                    "--rms-norm-eps",
                    f"{rms_norm_eps:.9g}",
                    "--output-f32",
                    str(token_output),
                    "--max-slot-mib",
                    str(max_slot_mib),
                    "--max-router-mib",
                    str(max_router_mib),
                    "--max-runner-scratch-mib",
                    str(max_runner_scratch_mib),
                ]
                if routed_scaling_factor is not None:
                    cmd.extend(
                        ["--routed-scaling-factor", f"{routed_scaling_factor:.9g}"]
                    )
                if norm_topk_prob:
                    cmd.append("--norm-topk-prob")
                if no_norm_topk_prob:
                    cmd.append("--no-norm-topk-prob")
                if router_n_group is not None:
                    cmd.extend(["--router-n-group", str(router_n_group)])
                if router_topk_group is not None:
                    cmd.extend(["--router-topk-group", str(router_topk_group)])
                if ignore_router_bias:
                    cmd.append("--ignore-router-bias")
                if include_shared_expert:
                    cmd.append("--include-shared-expert")
                if expert_read_advise_merge_gap_kib:
                    cmd.extend(
                        [
                            "--expert-read-advise-merge-gap-kib",
                            str(expert_read_advise_merge_gap_kib),
                        ]
                    )
                if expert_read_advise_align_kib:
                    cmd.extend(
                        [
                            "--expert-read-advise-align-kib",
                            str(expert_read_advise_align_kib),
                        ]
                    )
                if router_dir is not None:
                    cmd.extend(
                        [
                            "--output-router-json",
                            str(router_dir / f"{token_stem}.router.json"),
                        ]
                    )
                if not first_command:
                    first_command = tuple(cmd)
                _run_command(cmd, echo_output=echo_runner_output)
                token_row = token_output.read_bytes()
                if len(token_row) != token_bytes:
                    raise PrefillExecuteError(
                        f"token output bytes {len(token_row)} do not match expected "
                        f"{token_bytes}"
                    )
                batch_out.write(token_row)
                if not keep_token_files:
                    try:
                        token_input.unlink()
                        token_output.unlink()
                    except OSError as exc:
                        raise PrefillExecuteError(
                            f"failed to remove routed MLP token temp file: {exc}"
                        ) from exc
            trailing = source.read(1)
            if trailing:
                raise PrefillExecuteError("routed MLP input has trailing bytes")
    except OSError as exc:
        raise PrefillExecuteError(f"failed during routed MLP batch streaming: {exc}") from exc

    try:
        output_bytes = output_path.stat().st_size
    except OSError as exc:
        raise PrefillExecuteError(f"failed to stat routed MLP output {output_path}: {exc}") from exc
    if output_bytes != expected_input_bytes:
        raise PrefillExecuteError(
            f"output bytes {output_bytes} do not match expected {expected_input_bytes}"
        )
    return PrefillRoutedMLPBlockBatchResult(
        runner_path=runner,
        expert_layout_path=expert_layout,
        resident_layout_path=resident_layout,
        input_path=input_path,
        output_dir=out_dir,
        output_path=output_path,
        router_json_dir=router_dir,
        layer=layer,
        batch_tokens=batch_tokens,
        hidden_dim=budget.hidden_dim,
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        token_input_bytes=token_bytes,
        token_output_bytes=token_bytes,
        top_k=top_k,
        max_k=max_k,
        router_score=router_score,
        include_shared_expert=include_shared_expert,
        rms_norm_eps=rms_norm_eps,
        estimated_peak_bytes=budget.estimated_peak_bytes,
        read_bytes=batch_tokens * budget.read_bytes_per_token,
        command_count=batch_tokens,
        first_command=first_command,
    )
