from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path


class PrefillBackendError(RuntimeError):
    """Raised when prefill backend inspection arguments are invalid."""


DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS = 5.0


def default_prefill_backend_probe_path() -> Path:
    return Path(__file__).resolve().parent.parent / "metal/prefill-backend-probe"


@dataclass(frozen=True)
class PrefillBackendCapability:
    sdk_path: Path | None
    metal_headers_available: bool
    metal4_headers_available: bool
    metal_tensor_headers_available: bool
    metal_tensor_int4_declared: bool
    metal4_machine_learning_declared: bool
    mps_graph_matmul_declared: bool
    mpp_tensor_ops_symbol_declared: bool
    host_probe_ran: bool
    host_probe_ok: bool
    mpp_compile_probe_ran: bool
    mpp_compile_probe_ok: bool | None
    mpp_compile_variant: str | None
    mpp_compile_error: str | None
    device_name: str | None
    supports_metal4_family: bool | None
    responds_new_mtl4_command_queue: bool | None
    responds_new_tensor: bool | None
    responds_tensor_size_align: bool | None
    responds_new_compiler: bool | None
    can_allocate_tiny_ml_tensor: bool | None
    tensor_error: str | None
    recommended_backend: str
    reasons: tuple[str, ...]
    mpp_compile_probe_requested: bool = False
    mpp_run_probe_requested: bool = False
    mpp_run_probe_ran: bool = False
    mpp_run_probe_ok: bool | None = None
    mpp_run_probe_error: str | None = None
    mpp_run_probe_max_abs_error: float | None = None
    mpp_run_probe_kernel_variant: str | None = None
    mpp_run_probe_shape: str | None = None
    mpp_run_probe_dtype: str | None = None
    mpp_run_probe_execution_path: str | None = None
    host_probe_requested: bool = False
    host_probe_path: Path | None = None
    host_probe_sha256: str | None = None
    host_probe_error: str | None = None
    mps_graph_probe_requested: bool = False
    mps_graph_probe_ran: bool = False
    mps_graph_probe_ok: bool | None = None
    mps_graph_probe_error: str | None = None
    probe_timeout_seconds: float = DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS

    @property
    def metal4_ml_runtime_available(self) -> bool:
        return (
            self.metal4_machine_learning_declared
            and self.metal_tensor_headers_available
            and self.host_probe_ok
            and self.supports_metal4_family is True
            and self.responds_new_mtl4_command_queue is True
            and self.responds_new_tensor is True
            and self.responds_tensor_size_align is True
            and self.responds_new_compiler is True
            and self.can_allocate_tiny_ml_tensor is True
        )

    @property
    def mps_graph_runtime_available(self) -> bool:
        if self.mps_graph_probe_requested:
            return (
                self.mps_graph_matmul_declared
                and self.host_probe_ok
                and self.mps_graph_probe_ran
                and self.mps_graph_probe_ok is True
            )
        return (
            self.mps_graph_matmul_declared
            and (not self.host_probe_requested or self.host_probe_ok)
        )

    @property
    def mpp_runtime_available(self) -> bool:
        return (
            self.mpp_tensor_ops_symbol_declared
            and self.mpp_compile_probe_ran
            and self.mpp_compile_probe_ok is True
            and (
                not self.mpp_run_probe_requested
                or (
                    self.mpp_run_probe_ran
                    and self.mpp_run_probe_ok is True
                )
            )
            and self.metal4_ml_runtime_available
        )

    @property
    def prefill_acceleration_runtimes(self) -> tuple[str, ...]:
        return prefill_acceleration_runtimes(self)

    @property
    def selectable_accelerated_prefill_backends(self) -> tuple[str, ...]:
        return selectable_accelerated_prefill_backends(self)

    @property
    def selectable_prefill_acceleration_available(self) -> bool:
        return bool(self.selectable_accelerated_prefill_backends)

    @property
    def validated_accelerated_prefill_backends(self) -> tuple[str, ...]:
        return validated_accelerated_prefill_backends(self)

    @property
    def validated_prefill_acceleration_available(self) -> bool:
        return bool(self.validated_accelerated_prefill_backends)

    @property
    def prefill_acceleration_runtime_gaps(self) -> tuple[dict[str, str], ...]:
        return prefill_acceleration_runtime_gaps(self)

    @property
    def prefill_neural_accelerator_status(self) -> dict[str, object]:
        return prefill_neural_accelerator_status(self)

    @property
    def suggested_prefill_acceleration_flags(self) -> dict[str, object] | None:
        return suggested_prefill_acceleration_flags(self)


@dataclass(frozen=True)
class PrefillAccelerationRequirement:
    required: bool
    ok: bool
    configured_backend: str
    mps_graph_runtime_available: bool | None
    mps_graph_probe_requested: bool | None
    mps_graph_probe_ran: bool | None
    mps_graph_probe_ok: bool | None
    mpp_runtime_available: bool | None
    prefill_acceleration_runtimes: tuple[str, ...]
    accelerated_backends: tuple[str, ...]
    reason: str
    reason_code: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _bool_capability_attr(capability: object, name: str) -> bool:
    value = getattr(capability, name, None)
    if value is None and name == "mps_graph_runtime_available":
        value = getattr(capability, "mps_graph_matmul_declared", False)
    return value is True


def prefill_acceleration_runtimes(capability: object) -> tuple[str, ...]:
    """Return hardware/OS prefill acceleration runtimes visible on this host."""

    runtimes: list[str] = []
    if _bool_capability_attr(capability, "mpp_runtime_available"):
        runtimes.append("mpp_tensor_ops_prefill")
    if _bool_capability_attr(capability, "mps_graph_runtime_available"):
        runtimes.append("mpsgraph-f32")
    return tuple(runtimes)


def selectable_accelerated_prefill_backends(capability: object) -> tuple[str, ...]:
    """Return accelerated backends that current LargerLM commands can select."""

    backends: list[str] = []
    if _bool_capability_attr(capability, "mps_graph_runtime_available"):
        backends.append("mpsgraph-f32")
    return tuple(backends)


def _mps_graph_runtime_probe_passed(capability: object) -> bool:
    return (
        getattr(capability, "mps_graph_probe_requested", None) is True
        and getattr(capability, "mps_graph_probe_ran", None) is True
        and getattr(capability, "mps_graph_probe_ok", None) is True
    )


def validated_accelerated_prefill_backends(
    capability: object,
) -> tuple[str, ...]:
    """Return selectable accelerated backends with runtime proof for hard gates."""

    backends: list[str] = []
    if (
        "mpsgraph-f32" in selectable_accelerated_prefill_backends(capability)
        and _mps_graph_runtime_probe_passed(capability)
    ):
        backends.append("mpsgraph-f32")
    return tuple(backends)


def prefill_acceleration_runtime_gaps(
    capability: object,
) -> tuple[dict[str, str], ...]:
    """Return visible acceleration runtimes that generation cannot select yet."""

    selectable = set(selectable_accelerated_prefill_backends(capability))
    gaps: list[dict[str, str]] = []
    if (
        "mpp_tensor_ops_prefill" in prefill_acceleration_runtimes(capability)
        and "mpp_tensor_ops_prefill" not in selectable
    ):
        gaps.append(
            {
                "runtime": "mpp_tensor_ops_prefill",
                "reason": (
                    "MPP tensor ops runtime is visible but no selectable MPP "
                    "prefill execution backend is implemented"
                ),
            }
        )
    return tuple(gaps)


def prefill_neural_accelerator_status(capability: object) -> dict[str, object]:
    """Return the current MPP/Metal ML prefill readiness boundary.

    The MPP tensor-op path is the planned GPU neural-accelerator route for large
    prefill GEMMs. Keep it separate from selectable generation backends until a
    real execution backend is wired.
    """

    mpp_runtime = _bool_capability_attr(capability, "mpp_runtime_available")
    metal4_ml = (
        _bool_capability_attr(capability, "metal4_ml_runtime_available")
        or mpp_runtime
    )
    mpp_symbols = (
        _bool_capability_attr(capability, "mpp_tensor_ops_symbol_declared")
        or mpp_runtime
    )
    compile_requested = bool(
        getattr(capability, "mpp_compile_probe_requested", False)
    )
    compile_ran = bool(getattr(capability, "mpp_compile_probe_ran", False))
    compile_ok = getattr(capability, "mpp_compile_probe_ok", None) is True
    run_requested = bool(getattr(capability, "mpp_run_probe_requested", False))
    run_ran = bool(getattr(capability, "mpp_run_probe_ran", False))
    run_ok = getattr(capability, "mpp_run_probe_ok", None) is True
    selectable = (
        "mpp_tensor_ops_prefill" in selectable_accelerated_prefill_backends(capability)
    )
    if selectable:
        status = "selectable"
        reason = "MPP tensor ops prefill backend is selectable"
    elif metal4_ml and not mpp_symbols:
        status = "missing_public_mpp_symbols"
        reason = "Metal 4 ML runtime is visible but public MPP symbols are missing"
    elif mpp_symbols and not metal4_ml:
        status = "metal4_ml_runtime_unavailable"
        reason = "MPP symbols are visible but Metal 4 ML runtime is unavailable"
    elif run_requested and run_ran and not run_ok:
        status = "run_probe_failed"
        reason = "MPP tensor ops run probe failed"
    elif mpp_runtime and run_ok:
        status = "runtime_executed_not_selectable"
        reason = (
            "MPP tensor ops runtime executed a tiny matmul but generation has "
            "no selectable MPP prefill backend yet"
        )
    elif mpp_runtime:
        status = "runtime_visible_not_selectable"
        reason = (
            "MPP tensor ops runtime is visible but generation has no selectable "
            "MPP prefill backend yet"
        )
    elif mpp_symbols and metal4_ml and not compile_requested:
        status = "compile_probe_required"
        reason = "MPP symbols and Metal 4 ML runtime are visible; compile probe not run"
    elif mpp_symbols and metal4_ml and compile_ran and not compile_ok:
        status = "compile_probe_failed"
        reason = "MPP tensor ops compile probe failed"
    else:
        status = "unavailable"
        reason = "MPP tensor ops prefill runtime is not available"

    return {
        "runtime": "mpp_tensor_ops_prefill",
        "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
        "status": status,
        "ready_for_generation": selectable,
        "runtime_visible": mpp_runtime,
        "selectable": selectable,
        "metal4_ml_runtime_available": metal4_ml,
        "mpp_tensor_ops_symbol_declared": mpp_symbols,
        "mpp_compile_probe_requested": compile_requested,
        "mpp_compile_probe_ran": compile_ran,
        "mpp_compile_probe_ok": (
            True if compile_ok else getattr(capability, "mpp_compile_probe_ok", None)
        ),
        "mpp_run_probe_requested": run_requested,
        "mpp_run_probe_ran": run_ran,
        "mpp_run_probe_ok": (
            True if run_ok else getattr(capability, "mpp_run_probe_ok", None)
        ),
        "mpp_run_probe_error": getattr(capability, "mpp_run_probe_error", None),
        "mpp_run_probe_max_abs_error": getattr(
            capability,
            "mpp_run_probe_max_abs_error",
            None,
        ),
        "mpp_run_probe_kernel_variant": getattr(
            capability,
            "mpp_run_probe_kernel_variant",
            None,
        ),
        "mpp_run_probe_shape": getattr(capability, "mpp_run_probe_shape", None),
        "mpp_run_probe_dtype": getattr(capability, "mpp_run_probe_dtype", None),
        "mpp_run_probe_execution_path": getattr(
            capability,
            "mpp_run_probe_execution_path",
            None,
        ),
        "recommended_backend": getattr(capability, "recommended_backend", None),
        "reason": reason,
    }


def suggested_prefill_acceleration_flags(
    capability: object,
    *,
    source: str = "prefill_backend_probe",
) -> dict[str, object] | None:
    """Return argv-style flags for the best currently selectable accelerated path."""

    selectable = selectable_accelerated_prefill_backends(capability)
    if not selectable:
        return None
    backend = selectable[0]
    runtime_probe_argv: tuple[str, ...] = ()
    runtime_probe_required = backend == "mpsgraph-f32"
    runtime_probe_satisfied = False
    if runtime_probe_required:
        runtime_probe_argv = ("--run-mpsgraph-probe",)
        runtime_probe_satisfied = (
            getattr(capability, "mps_graph_probe_requested", False) is True
            and getattr(capability, "mps_graph_probe_ran", False) is True
            and getattr(capability, "mps_graph_probe_ok", None) is True
        )
    return {
        "source": source,
        "prefill_linear_backend": backend,
        "prefill_acceleration_runtimes": prefill_acceleration_runtimes(capability),
        "selectable_accelerated_prefill_backends": selectable,
        "validated_accelerated_prefill_backends": (
            validated_accelerated_prefill_backends(capability)
        ),
        "prefill_acceleration_runtime_gaps": (
            prefill_acceleration_runtime_gaps(capability)
        ),
        "prefill_neural_accelerator_status": (
            prefill_neural_accelerator_status(capability)
        ),
        "runtime_probe_backend": backend if runtime_probe_required else None,
        "runtime_probe_required": runtime_probe_required,
        "runtime_probe_satisfied": runtime_probe_satisfied,
        "runtime_probe_argv": runtime_probe_argv,
        "argv": (
            "--prefill-linear-backend",
            backend,
            "--require-prefill-acceleration",
            *runtime_probe_argv,
        ),
    }


def evaluate_prefill_acceleration_requirement(
    *,
    configured_backend: str,
    mps_graph_runtime_available: bool | None,
    mpp_runtime_available: bool | None,
    mps_graph_probe_requested: bool | None = None,
    mps_graph_probe_ran: bool | None = None,
    mps_graph_probe_ok: bool | None = None,
    selectable_backends: tuple[str, ...] = (),
    acceleration_runtimes: tuple[str, ...] = (),
) -> PrefillAccelerationRequirement:
    accelerated = tuple(selectable_backends)
    runtimes = tuple(acceleration_runtimes)
    mpsgraph_probe_passed = (
        mps_graph_probe_requested is True
        and mps_graph_probe_ran is True
        and mps_graph_probe_ok is True
    )
    if configured_backend == "custom-metal":
        ok = False
        reason = "custom-metal was configured"
        reason_code = "custom_metal_configured"
    elif configured_backend == "mpsgraph-f32":
        if mps_graph_runtime_available is not True:
            ok = False
            reason = "mpsgraph-f32 was configured but MPSGraph runtime is unavailable"
            reason_code = "mpsgraph_runtime_unavailable"
        elif not mpsgraph_probe_passed:
            ok = False
            reason = (
                "mpsgraph-f32 requires --run-mpsgraph-probe and a passing "
                "runtime probe"
            )
            reason_code = "mpsgraph_runtime_probe_required"
        else:
            ok = True
            reason = ""
            reason_code = "ok"
    elif configured_backend == "mps-matrix-f32":
        ok = True
        reason = ""
        reason_code = "ok"
        if "mps-matrix-f32" not in accelerated:
            accelerated = (*accelerated, "mps-matrix-f32")
    else:
        ok = bool(accelerated)
        if ok and "mpsgraph-f32" in accelerated and not mpsgraph_probe_passed:
            ok = False
            reason = (
                "selectable mpsgraph-f32 requires --run-mpsgraph-probe and a "
                "passing runtime probe"
            )
            reason_code = "selectable_mpsgraph_runtime_probe_required"
        elif ok:
            reason = ""
            reason_code = "ok"
        elif mpp_runtime_available is True:
            reason = (
                "MPP tensor ops runtime is visible but no selectable MPP "
                "prefill backend is implemented"
            )
            reason_code = "mpp_runtime_not_selectable"
        else:
            reason = "no accelerated prefill backend is available"
            reason_code = "no_accelerated_backend"
    return PrefillAccelerationRequirement(
        required=True,
        ok=ok,
        configured_backend=configured_backend,
        mps_graph_runtime_available=mps_graph_runtime_available,
        mps_graph_probe_requested=mps_graph_probe_requested,
        mps_graph_probe_ran=mps_graph_probe_ran,
        mps_graph_probe_ok=mps_graph_probe_ok,
        mpp_runtime_available=mpp_runtime_available,
        prefill_acceleration_runtimes=runtimes,
        accelerated_backends=accelerated,
        reason=reason,
        reason_code=reason_code,
    )


def _run_text(args: list[str], *, timeout_seconds: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception:
        return None
    return result.stdout.strip()


def find_macos_sdk_path() -> Path | None:
    value = _run_text(["xcrun", "--show-sdk-path"])
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def _contains_any(paths: list[Path], needles: tuple[str, ...]) -> bool:
    for path in paths:
        text = _read_text(path)
        if any(needle in text for needle in needles):
            return True
    return False


def _metal_header(sdk_path: Path, name: str) -> Path:
    return (
        sdk_path
        / "System/Library/Frameworks/Metal.framework/Versions/A/Headers"
        / name
    )


def _mps_graph_header(sdk_path: Path, name: str) -> Path:
    return (
        sdk_path
        / "System/Library/Frameworks/MetalPerformanceShadersGraph.framework"
        / "Versions/A/Headers"
        / name
    )


def _file_sha256(path: Path) -> str | None:
    digest = sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _load_probe(
    path: Path,
    *,
    timeout_seconds: float,
    compile_mpp_probe: bool,
    run_mpp_probe: bool,
    run_mpsgraph_probe: bool,
) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, "probe binary does not exist"
    args = [str(path)]
    if compile_mpp_probe:
        args.append("--compile-mpp")
    if run_mpp_probe:
        args.append("--run-mpp-probe")
    if run_mpsgraph_probe:
        args.append("--run-mpsgraph-probe")
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout_seconds,
        )
        payload = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return None, f"probe timed out after {timeout_seconds:g}s"
    except subprocess.CalledProcessError as exc:
        detail = _compact_text(exc.stderr) or _compact_text(exc.stdout)
        suffix = f": {detail}" if detail else ""
        return None, f"probe exited with status {exc.returncode}{suffix}"
    except OSError as exc:
        return None, f"failed to run probe: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"probe emitted invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "probe JSON must be an object"
    return payload, None


def _bool_or_none(payload: dict[str, object] | None, key: str) -> bool | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    return None


def _str_or_none(payload: dict[str, object] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _float_or_none(payload: dict[str, object] | None, key: str) -> float | None:
    if payload is None:
        return None
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return None


def _compact_text(value: str | None, *, max_chars: int = 240) -> str | None:
    if not value:
        return None
    compact = " ".join(value.strip().split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def inspect_prefill_backend(
    *,
    sdk_path: str | Path | None = None,
    probe_binary: str | Path | None = None,
    run_host_probe: bool = True,
    compile_mpp_probe: bool = False,
    run_mpp_probe: bool = False,
    run_mpsgraph_probe: bool = False,
    probe_timeout_seconds: float = DEFAULT_PREFILL_BACKEND_PROBE_TIMEOUT_SECONDS,
) -> PrefillBackendCapability:
    if isinstance(probe_timeout_seconds, bool):
        raise PrefillBackendError("probe_timeout_seconds must be positive")
    try:
        probe_timeout_seconds = float(probe_timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise PrefillBackendError("probe_timeout_seconds must be positive") from exc
    if not math.isfinite(probe_timeout_seconds) or probe_timeout_seconds <= 0:
        raise PrefillBackendError("probe_timeout_seconds must be positive")
    if compile_mpp_probe and not run_host_probe:
        raise PrefillBackendError("compile_mpp_probe requires run_host_probe")
    if run_mpp_probe and not run_host_probe:
        raise PrefillBackendError("run_mpp_probe requires run_host_probe")
    if run_mpsgraph_probe and not run_host_probe:
        raise PrefillBackendError("run_mpsgraph_probe requires run_host_probe")
    compile_mpp_probe = bool(compile_mpp_probe or run_mpp_probe)

    sdk = Path(sdk_path) if sdk_path is not None else find_macos_sdk_path()
    if sdk is not None and not sdk.exists():
        raise PrefillBackendError(f"SDK path does not exist: {sdk}")

    metal_h = _metal_header(sdk, "Metal.h") if sdk else Path()
    tensor_h = _metal_header(sdk, "MTLTensor.h") if sdk else Path()
    ml_pipeline_h = _metal_header(sdk, "MTL4MachineLearningPipeline.h") if sdk else Path()
    ml_encoder_h = (
        _metal_header(sdk, "MTL4MachineLearningCommandEncoder.h") if sdk else Path()
    )
    command_buffer_h = _metal_header(sdk, "MTL4CommandBuffer.h") if sdk else Path()
    compiler_h = _metal_header(sdk, "MTL4Compiler.h") if sdk else Path()
    device_h = _metal_header(sdk, "MTLDevice.h") if sdk else Path()
    mps_matmul_h = (
        _mps_graph_header(sdk, "MPSGraphMatrixMultiplicationOps.h") if sdk else Path()
    )

    metal_headers_available = metal_h.exists()
    metal_tensor_headers_available = tensor_h.exists()
    metal4_headers_available = all(
        path.exists()
        for path in (ml_pipeline_h, ml_encoder_h, command_buffer_h, compiler_h)
    )
    metal_tensor_int4_declared = _contains_any(
        [tensor_h],
        ("MTLTensorDataTypeInt4", "MTLTensorDataTypeUInt4"),
    )
    metal4_machine_learning_declared = _contains_any(
        [ml_pipeline_h, ml_encoder_h, command_buffer_h, compiler_h],
        (
            "MTL4MachineLearningPipelineDescriptor",
            "MTL4MachineLearningCommandEncoder",
            "newMachineLearningPipelineStateWithDescriptor",
            "machineLearningCommandEncoder",
        ),
    )
    mps_graph_matmul_declared = _contains_any(
        [mps_matmul_h],
        ("matrixMultiplication", "MPSGraphMatrixMultiplication"),
    )

    search_paths = [
        metal_h,
        tensor_h,
        ml_pipeline_h,
        ml_encoder_h,
        command_buffer_h,
        compiler_h,
        device_h,
        mps_matmul_h,
    ]
    mpp_tensor_ops_symbol_declared = _contains_any(
        search_paths,
        ("mpp::tensor_ops", "namespace mpp", "tensor_ops"),
    )

    default_probe = default_prefill_backend_probe_path()
    probe_path = Path(probe_binary) if probe_binary is not None else default_probe
    probe_payload: dict[str, object] | None = None
    host_probe_error: str | None = None
    if run_host_probe:
        probe_payload, host_probe_error = _load_probe(
            probe_path,
            timeout_seconds=probe_timeout_seconds,
            compile_mpp_probe=compile_mpp_probe,
            run_mpp_probe=run_mpp_probe,
            run_mpsgraph_probe=run_mpsgraph_probe,
        )
    host_probe_ran = run_host_probe and probe_payload is not None
    host_probe_ok = _bool_or_none(probe_payload, "ok") is True
    device_name = _str_or_none(probe_payload, "device_name")
    tensor_error = _str_or_none(probe_payload, "tensor_error")
    mps_graph_probe_ran = bool(
        run_mpsgraph_probe
        and host_probe_ran
        and _bool_or_none(probe_payload, "mps_graph_probe_ran")
    )
    mps_graph_probe_ok = _bool_or_none(probe_payload, "mps_graph_probe_ok")
    mps_graph_probe_error = _str_or_none(probe_payload, "mps_graph_probe_error")

    reasons: list[str] = []
    if sdk is None:
        reasons.append("macOS SDK path was not discovered with xcrun")
    if not metal_headers_available:
        reasons.append("Metal headers are not present in the selected SDK")
    if not metal4_headers_available:
        reasons.append("Metal 4 machine-learning headers are not complete")
    if not metal_tensor_headers_available:
        reasons.append("MTLTensor headers are not present")
    if metal_tensor_headers_available and not metal_tensor_int4_declared:
        reasons.append("MTLTensor int4/uint4 datatypes are not declared")
    if metal4_headers_available and not metal4_machine_learning_declared:
        reasons.append("Metal 4 machine-learning API symbols are not declared")
    if not mpp_tensor_ops_symbol_declared:
        reasons.append("mpp::tensor_ops symbols were not found in public SDK headers")
    if run_host_probe and not host_probe_ran:
        detail = f": {host_probe_error}" if host_probe_error else ""
        reasons.append(f"host probe was not run successfully: {probe_path}{detail}")
    if compile_mpp_probe and host_probe_ran:
        if _bool_or_none(probe_payload, "mpp_compile_probe_ok") is not True:
            reasons.append("runtime MPP matmul2d compile probe failed")
    mpp_run_probe_ran = bool(
        run_mpp_probe
        and host_probe_ran
        and _bool_or_none(probe_payload, "mpp_run_probe_ran")
    )
    mpp_run_probe_ok = _bool_or_none(probe_payload, "mpp_run_probe_ok")
    mpp_run_probe_error = _str_or_none(probe_payload, "mpp_run_probe_error")
    if run_mpp_probe and host_probe_ran:
        if mpp_run_probe_ok is not True:
            reasons.append("runtime MPP matmul2d run probe failed")
            compact_mpp_run_error = _compact_text(mpp_run_probe_error)
            if compact_mpp_run_error is not None:
                reasons.append(f"MPP matmul2d run probe failed: {compact_mpp_run_error}")
    if run_mpsgraph_probe and host_probe_ran:
        if mps_graph_probe_ok is not True:
            reasons.append("runtime MPSGraph matmul probe failed")
            compact_mps_graph_error = _compact_text(mps_graph_probe_error)
            if compact_mps_graph_error is not None:
                reasons.append(f"MPSGraph matmul probe failed: {compact_mps_graph_error}")

    supports_metal4 = _bool_or_none(probe_payload, "supports_metal4_family")
    responds_queue = _bool_or_none(probe_payload, "responds_new_mtl4_command_queue")
    responds_tensor = _bool_or_none(probe_payload, "responds_new_tensor")
    responds_tensor_size = _bool_or_none(probe_payload, "responds_tensor_size_align")
    responds_compiler = _bool_or_none(probe_payload, "responds_new_compiler")
    can_allocate_tiny_ml_tensor = _bool_or_none(
        probe_payload,
        "can_allocate_tiny_ml_tensor",
    )
    if host_probe_ran and host_probe_ok is not True:
        if tensor_error == "no Metal device" or device_name is None:
            reasons.append("host probe could not create a default Metal device")
        else:
            reasons.append("host probe did not report Metal runtime readiness")
    if host_probe_ran and host_probe_ok and supports_metal4 is not True:
        reasons.append("default Metal device does not report MTLGPUFamilyMetal4")
    if host_probe_ran and host_probe_ok and responds_queue is not True:
        reasons.append("default Metal device lacks newMTL4CommandQueue")
    if host_probe_ran and host_probe_ok and responds_tensor is not True:
        reasons.append("default Metal device lacks newTensorWithDescriptor:error:")
    if host_probe_ran and host_probe_ok and responds_tensor_size is not True:
        reasons.append("default Metal device lacks tensorSizeAndAlignWithDescriptor:")
    if host_probe_ran and host_probe_ok and responds_compiler is not True:
        reasons.append("default Metal device lacks newCompilerWithDescriptor:")
    if host_probe_ran and can_allocate_tiny_ml_tensor is not True:
        reasons.append("default Metal device cannot allocate a tiny ML tensor")
        compact_tensor_error = _compact_text(tensor_error)
        if compact_tensor_error is not None:
            reasons.append(f"tiny ML tensor allocation failed: {compact_tensor_error}")

    mpp_compile_probe_ran = bool(
        compile_mpp_probe
        and host_probe_ran
        and _bool_or_none(probe_payload, "mpp_compile_probe_ran")
    )
    metal4_ml_runtime_ready = (
        metal4_machine_learning_declared
        and metal_tensor_headers_available
        and host_probe_ok
        and supports_metal4 is True
        and responds_queue is True
        and responds_tensor is True
        and responds_tensor_size is True
        and responds_compiler is True
        and can_allocate_tiny_ml_tensor is True
    )
    mpp_ready = (
        mpp_tensor_ops_symbol_declared
        and mpp_compile_probe_ran
        and _bool_or_none(probe_payload, "mpp_compile_probe_ok") is True
        and (not run_mpp_probe or mpp_run_probe_ok is True)
        and metal4_ml_runtime_ready
    )
    mpsgraph_ready = mps_graph_matmul_declared and (
        (
            host_probe_ok
            and mps_graph_probe_ran
            and mps_graph_probe_ok is True
        )
        if run_mpsgraph_probe
        else (not run_host_probe or host_probe_ok)
    )
    if mpp_ready:
        recommended = "mpp_tensor_ops_prefill"
    elif mpsgraph_ready:
        recommended = "mpsgraph_prefill_fallback"
    else:
        recommended = "custom_metal_prefill_fallback"

    return PrefillBackendCapability(
        sdk_path=sdk,
        metal_headers_available=metal_headers_available,
        metal4_headers_available=metal4_headers_available,
        metal_tensor_headers_available=metal_tensor_headers_available,
        metal_tensor_int4_declared=metal_tensor_int4_declared,
        metal4_machine_learning_declared=metal4_machine_learning_declared,
        mps_graph_matmul_declared=mps_graph_matmul_declared,
        mpp_tensor_ops_symbol_declared=mpp_tensor_ops_symbol_declared,
        host_probe_ran=host_probe_ran,
        host_probe_ok=host_probe_ok,
        mpp_compile_probe_ran=mpp_compile_probe_ran,
        mpp_compile_probe_ok=_bool_or_none(probe_payload, "mpp_compile_probe_ok"),
        mpp_compile_variant=_str_or_none(probe_payload, "mpp_compile_variant"),
        mpp_compile_error=_compact_text(_str_or_none(probe_payload, "mpp_compile_error")),
        device_name=device_name,
        supports_metal4_family=supports_metal4,
        responds_new_mtl4_command_queue=responds_queue,
        responds_new_tensor=responds_tensor,
        responds_tensor_size_align=responds_tensor_size,
        responds_new_compiler=responds_compiler,
        can_allocate_tiny_ml_tensor=can_allocate_tiny_ml_tensor,
        tensor_error=tensor_error,
        recommended_backend=recommended,
        reasons=tuple(reasons),
        mpp_compile_probe_requested=bool(compile_mpp_probe),
        mpp_run_probe_requested=bool(run_mpp_probe),
        mpp_run_probe_ran=mpp_run_probe_ran,
        mpp_run_probe_ok=mpp_run_probe_ok,
        mpp_run_probe_error=_compact_text(mpp_run_probe_error),
        mpp_run_probe_max_abs_error=_float_or_none(
            probe_payload,
            "mpp_run_probe_max_abs_error",
        ),
        mpp_run_probe_kernel_variant=_str_or_none(
            probe_payload,
            "mpp_run_probe_kernel_variant",
        ),
        mpp_run_probe_shape=_str_or_none(probe_payload, "mpp_run_probe_shape"),
        mpp_run_probe_dtype=_str_or_none(probe_payload, "mpp_run_probe_dtype"),
        mpp_run_probe_execution_path=_str_or_none(
            probe_payload,
            "mpp_run_probe_execution_path",
        ),
        host_probe_requested=bool(run_host_probe),
        host_probe_path=probe_path if run_host_probe else None,
        host_probe_sha256=_file_sha256(probe_path) if run_host_probe else None,
        host_probe_error=host_probe_error,
        mps_graph_probe_requested=bool(run_mpsgraph_probe),
        mps_graph_probe_ran=mps_graph_probe_ran,
        mps_graph_probe_ok=mps_graph_probe_ok,
        mps_graph_probe_error=mps_graph_probe_error,
        probe_timeout_seconds=probe_timeout_seconds,
    )
