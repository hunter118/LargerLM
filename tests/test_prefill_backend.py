from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from largerlm.cli import main as cli_main
from largerlm.prefill_backend import (
    PrefillBackendError,
    evaluate_prefill_acceleration_requirement,
    inspect_prefill_backend,
    prefill_acceleration_runtime_gaps,
    prefill_acceleration_runtimes,
    prefill_neural_accelerator_status,
    selectable_accelerated_prefill_backends,
    suggested_prefill_acceleration_flags,
    validated_accelerated_prefill_backends,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_evaluate_prefill_acceleration_requirement_accepts_mpsgraph_probe() -> None:
    gate = evaluate_prefill_acceleration_requirement(
        configured_backend="auto",
        mps_graph_runtime_available=True,
        mpp_runtime_available=False,
        mps_graph_probe_requested=True,
        mps_graph_probe_ran=True,
        mps_graph_probe_ok=True,
        selectable_backends=("mpsgraph-f32",),
        acceleration_runtimes=("mpsgraph-f32",),
    )

    assert gate.ok is True
    assert gate.reason_code == "ok"
    assert gate.accelerated_backends == ("mpsgraph-f32",)


def test_evaluate_prefill_acceleration_requirement_requires_mpsgraph_probe() -> None:
    gate = evaluate_prefill_acceleration_requirement(
        configured_backend="auto",
        mps_graph_runtime_available=True,
        mpp_runtime_available=False,
        mps_graph_probe_requested=False,
        mps_graph_probe_ran=False,
        mps_graph_probe_ok=None,
        selectable_backends=("mpsgraph-f32",),
        acceleration_runtimes=("mpsgraph-f32",),
    )

    assert gate.ok is False
    assert gate.reason_code == "selectable_mpsgraph_runtime_probe_required"
    assert "selectable mpsgraph-f32 requires" in gate.reason


def test_evaluate_prefill_acceleration_requirement_accepts_forced_mps_matrix() -> None:
    gate = evaluate_prefill_acceleration_requirement(
        configured_backend="mps-matrix-f32",
        mps_graph_runtime_available=False,
        mpp_runtime_available=False,
        selectable_backends=(),
        acceleration_runtimes=(),
    )

    assert gate.ok is True
    assert gate.reason_code == "ok"
    assert gate.accelerated_backends == ("mps-matrix-f32",)


def test_evaluate_prefill_acceleration_requirement_reports_mpp_gap() -> None:
    gate = evaluate_prefill_acceleration_requirement(
        configured_backend="auto",
        mps_graph_runtime_available=False,
        mpp_runtime_available=True,
        selectable_backends=(),
        acceleration_runtimes=("mpp_tensor_ops_prefill",),
    )

    assert gate.ok is False
    assert gate.reason_code == "mpp_runtime_not_selectable"
    assert "no selectable MPP prefill backend" in gate.reason


def _write_fake_sdk_header(sdk: Path, relative: str, text: str) -> None:
    path = sdk / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _fake_sdk(tmp_path: Path, *, include_mpp_symbol: bool = False) -> Path:
    sdk = tmp_path / "MacOSX.sdk"
    metal = "System/Library/Frameworks/Metal.framework/Versions/A/Headers"
    mps = (
        "System/Library/Frameworks/MetalPerformanceShadersGraph.framework/"
        "Versions/A/Headers"
    )
    _write_fake_sdk_header(sdk, f"{metal}/Metal.h", "#import <Metal/MTLTensor.h>\n")
    _write_fake_sdk_header(
        sdk,
        f"{metal}/MTLTensor.h",
        "MTLTensorDataTypeInt4 MTLTensorDataTypeUInt4 MTLTensorUsageMachineLearning\n",
    )
    _write_fake_sdk_header(
        sdk,
        f"{metal}/MTL4MachineLearningPipeline.h",
        "MTL4MachineLearningPipelineDescriptor\n",
    )
    _write_fake_sdk_header(
        sdk,
        f"{metal}/MTL4MachineLearningCommandEncoder.h",
        "MTL4MachineLearningCommandEncoder\n",
    )
    _write_fake_sdk_header(
        sdk,
        f"{metal}/MTL4CommandBuffer.h",
        "machineLearningCommandEncoder\n",
    )
    _write_fake_sdk_header(
        sdk,
        f"{metal}/MTL4Compiler.h",
        "newMachineLearningPipelineStateWithDescriptor\n",
    )
    mpp = " namespace mpp { namespace tensor_ops {} }\n" if include_mpp_symbol else ""
    _write_fake_sdk_header(sdk, f"{metal}/MTLDevice.h", f"MTLGPUFamilyMetal4\n{mpp}")
    _write_fake_sdk_header(
        sdk,
        f"{mps}/MPSGraphMatrixMultiplicationOps.h",
        "matrixMultiplication MPSGraphMatrixMultiplication\n",
    )
    return sdk


def _write_probe(path: Path, payload: dict[str, object]) -> Path:
    text = json.dumps(payload)
    path.write_text(
        "#!/usr/bin/env python3\n"
        f"print({text!r})\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)
    return path


def _write_arg_sensitive_probe(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "run_mpsgraph = '--run-mpsgraph-probe' in sys.argv\n"
        "run_mpp = '--run-mpp-probe' in sys.argv\n"
        "compile_mpp = '--compile-mpp' in sys.argv\n"
        "print(json.dumps({\n"
        "    'ok': True,\n"
        "    'device_name': 'Test GPU',\n"
        "    'supports_metal4_family': False,\n"
        "    'responds_new_mtl4_command_queue': False,\n"
        "    'responds_new_tensor': False,\n"
        "    'responds_tensor_size_align': False,\n"
        "    'responds_new_compiler': False,\n"
        "    'can_allocate_tiny_ml_tensor': False,\n"
        "    'mpp_compile_probe_ran': compile_mpp or run_mpp,\n"
        "    'mpp_compile_probe_ok': True if run_mpp else (False if compile_mpp else None),\n"
        "    'mpp_compile_variant': 'fake_mpp' if run_mpp else '',\n"
        "    'mpp_compile_error': '',\n"
        "    'mpp_run_probe_ran': run_mpp,\n"
        "    'mpp_run_probe_ok': True if run_mpp else None,\n"
        "    'mpp_run_probe_error': '',\n"
        "    'mpp_run_probe_max_abs_error': 0.0 if run_mpp else None,\n"
        "    'mpp_run_probe_kernel_variant': 'fake_mpp' if run_mpp else None,\n"
        "    'mpp_run_probe_shape': '32x32x32' if run_mpp else None,\n"
        "    'mpp_run_probe_dtype': 'half' if run_mpp else None,\n"
        "    'mpp_run_probe_execution_path': 'mpp::tensor_ops::matmul2d' if run_mpp else None,\n"
        "    'mps_graph_probe_ran': run_mpsgraph,\n"
        "    'mps_graph_probe_ok': True if run_mpsgraph else None,\n"
        "    'mps_graph_probe_error': '',\n"
        "}))\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)
    return path


def test_prefill_backend_inspects_metal4_headers(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path)

    capability = inspect_prefill_backend(sdk_path=sdk, run_host_probe=False)

    assert capability.metal4_headers_available
    assert capability.metal_tensor_headers_available
    assert capability.metal_tensor_int4_declared
    assert capability.metal4_machine_learning_declared
    assert capability.mps_graph_matmul_declared
    assert capability.mps_graph_runtime_available
    assert not capability.mpp_tensor_ops_symbol_declared
    assert capability.recommended_backend == "mpsgraph_prefill_fallback"
    assert capability.prefill_acceleration_runtimes == ("mpsgraph-f32",)
    assert capability.selectable_accelerated_prefill_backends == ("mpsgraph-f32",)
    assert capability.selectable_prefill_acceleration_available is True
    assert capability.validated_accelerated_prefill_backends == ()
    assert capability.validated_prefill_acceleration_available is False
    assert capability.prefill_acceleration_runtime_gaps == ()
    assert capability.suggested_prefill_acceleration_flags == {
        "source": "prefill_backend_probe",
        "prefill_linear_backend": "mpsgraph-f32",
        "prefill_acceleration_runtimes": ("mpsgraph-f32",),
        "selectable_accelerated_prefill_backends": ("mpsgraph-f32",),
        "validated_accelerated_prefill_backends": (),
        "prefill_acceleration_runtime_gaps": (),
        "prefill_neural_accelerator_status": (
            capability.prefill_neural_accelerator_status
        ),
        "runtime_probe_backend": "mpsgraph-f32",
        "runtime_probe_required": True,
        "runtime_probe_satisfied": False,
        "runtime_probe_argv": ("--run-mpsgraph-probe",),
        "argv": (
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
        ),
    }


def test_prefill_backend_can_require_mpsgraph_runtime_probe(
    tmp_path: Path,
) -> None:
    sdk = _fake_sdk(tmp_path)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "device_name": "Test GPU",
            "mps_graph_probe_ran": True,
            "mps_graph_probe_ok": True,
            "mps_graph_probe_error": "",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        run_mpsgraph_probe=True,
    )

    assert capability.mps_graph_probe_requested is True
    assert capability.mps_graph_probe_ran is True
    assert capability.mps_graph_probe_ok is True
    assert capability.mps_graph_probe_error is None
    assert capability.mps_graph_runtime_available is True
    assert capability.recommended_backend == "mpsgraph_prefill_fallback"
    assert capability.prefill_acceleration_runtimes == ("mpsgraph-f32",)
    assert capability.validated_accelerated_prefill_backends == ("mpsgraph-f32",)
    assert capability.validated_prefill_acceleration_available is True
    assert capability.suggested_prefill_acceleration_flags is not None
    assert (
        capability.suggested_prefill_acceleration_flags["runtime_probe_satisfied"]
        is True
    )
    assert "--run-mpsgraph-probe" in (
        capability.suggested_prefill_acceleration_flags["argv"]
    )


def test_prefill_backend_mpsgraph_probe_failure_disables_runtime_when_requested(
    tmp_path: Path,
) -> None:
    sdk = _fake_sdk(tmp_path)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "device_name": "Test GPU",
            "mps_graph_probe_ran": True,
            "mps_graph_probe_ok": False,
            "mps_graph_probe_error": "MPSGraph execution failed",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        run_mpsgraph_probe=True,
    )

    assert capability.mps_graph_probe_requested is True
    assert capability.mps_graph_probe_ran is True
    assert capability.mps_graph_probe_ok is False
    assert capability.mps_graph_probe_error == "MPSGraph execution failed"
    assert capability.mps_graph_runtime_available is False
    assert capability.recommended_backend == "custom_metal_prefill_fallback"
    assert "runtime MPSGraph matmul probe failed" in capability.reasons
    assert "MPSGraph matmul probe failed: MPSGraph execution failed" in (
        capability.reasons
    )
    assert capability.prefill_acceleration_runtimes == ()


def test_prefill_backend_mpsgraph_probe_requires_host_probe(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path)

    with pytest.raises(PrefillBackendError, match="run_mpsgraph_probe"):
        inspect_prefill_backend(
            sdk_path=sdk,
            run_host_probe=False,
            run_mpsgraph_probe=True,
        )


def test_prefill_backend_mpp_compile_probe_requires_host_probe(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)

    with pytest.raises(PrefillBackendError, match="compile_mpp_probe"):
        inspect_prefill_backend(
            sdk_path=sdk,
            run_host_probe=False,
            compile_mpp_probe=True,
        )


def test_prefill_backend_mpp_run_probe_requires_host_probe(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)

    with pytest.raises(PrefillBackendError, match="run_mpp_probe"):
        inspect_prefill_backend(
            sdk_path=sdk,
            run_host_probe=False,
            run_mpp_probe=True,
        )


def test_prefill_backend_reports_mpp_symbol_when_present(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)

    capability = inspect_prefill_backend(sdk_path=sdk, run_host_probe=False)

    assert capability.mpp_tensor_ops_symbol_declared
    assert not capability.mpp_runtime_available
    assert capability.recommended_backend == "mpsgraph_prefill_fallback"


def test_prefill_backend_recommends_mpp_after_compile_probe(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "supports_metal4_family": True,
            "responds_new_mtl4_command_queue": True,
            "responds_new_tensor": True,
            "responds_tensor_size_align": True,
            "responds_new_compiler": True,
            "can_allocate_tiny_ml_tensor": True,
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": True,
            "mpp_compile_variant": "metal_mpp",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        compile_mpp_probe=True,
    )

    assert capability.mpp_runtime_available
    assert capability.mpp_compile_probe_requested is True
    assert capability.responds_tensor_size_align is True
    assert capability.recommended_backend == "mpp_tensor_ops_prefill"
    assert capability.prefill_acceleration_runtimes == (
        "mpp_tensor_ops_prefill",
        "mpsgraph-f32",
    )
    assert capability.selectable_accelerated_prefill_backends == ("mpsgraph-f32",)
    assert capability.prefill_acceleration_runtime_gaps == (
        {
            "runtime": "mpp_tensor_ops_prefill",
            "reason": (
                "MPP tensor ops runtime is visible but no selectable MPP "
                "prefill execution backend is implemented"
            ),
        },
    )
    assert capability.prefill_neural_accelerator_status == {
        "runtime": "mpp_tensor_ops_prefill",
        "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
        "status": "runtime_visible_not_selectable",
        "ready_for_generation": False,
        "runtime_visible": True,
        "selectable": False,
        "metal4_ml_runtime_available": True,
        "mpp_tensor_ops_symbol_declared": True,
        "mpp_compile_probe_requested": True,
        "mpp_compile_probe_ran": True,
        "mpp_compile_probe_ok": True,
        "mpp_run_probe_requested": False,
        "mpp_run_probe_ran": False,
        "mpp_run_probe_ok": None,
        "mpp_run_probe_error": None,
        "mpp_run_probe_max_abs_error": None,
        "mpp_run_probe_kernel_variant": None,
        "mpp_run_probe_shape": None,
        "mpp_run_probe_dtype": None,
        "mpp_run_probe_execution_path": None,
        "recommended_backend": "mpp_tensor_ops_prefill",
        "reason": (
            "MPP tensor ops runtime is visible but generation has no selectable "
            "MPP prefill backend yet"
        ),
    }


def test_prefill_backend_can_run_mpp_probe(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "supports_metal4_family": True,
            "responds_new_mtl4_command_queue": True,
            "responds_new_tensor": True,
            "responds_tensor_size_align": True,
            "responds_new_compiler": True,
            "can_allocate_tiny_ml_tensor": True,
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": True,
            "mpp_compile_variant": "metal_mpp",
            "mpp_run_probe_ran": True,
            "mpp_run_probe_ok": True,
            "mpp_run_probe_error": "",
            "mpp_run_probe_max_abs_error": 0.0,
            "mpp_run_probe_kernel_variant": "metal_mpp",
            "mpp_run_probe_shape": "32x32x32",
            "mpp_run_probe_dtype": "half",
            "mpp_run_probe_execution_path": "mpp::tensor_ops::matmul2d",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        run_mpp_probe=True,
    )

    assert capability.mpp_compile_probe_requested is True
    assert capability.mpp_run_probe_requested is True
    assert capability.mpp_run_probe_ran is True
    assert capability.mpp_run_probe_ok is True
    assert capability.mpp_run_probe_max_abs_error == 0.0
    assert capability.mpp_run_probe_kernel_variant == "metal_mpp"
    assert capability.mpp_run_probe_shape == "32x32x32"
    assert capability.mpp_run_probe_dtype == "half"
    assert capability.mpp_run_probe_execution_path == "mpp::tensor_ops::matmul2d"
    assert capability.mpp_runtime_available is True
    assert capability.recommended_backend == "mpp_tensor_ops_prefill"
    assert capability.prefill_neural_accelerator_status["status"] == (
        "runtime_executed_not_selectable"
    )
    assert capability.prefill_neural_accelerator_status[
        "mpp_run_probe_shape"
    ] == "32x32x32"


def test_prefill_backend_run_mpp_probe_failure_disables_runtime(
    tmp_path: Path,
) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "supports_metal4_family": True,
            "responds_new_mtl4_command_queue": True,
            "responds_new_tensor": True,
            "responds_tensor_size_align": True,
            "responds_new_compiler": True,
            "can_allocate_tiny_ml_tensor": True,
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": True,
            "mpp_compile_variant": "metal_mpp",
            "mpp_run_probe_ran": True,
            "mpp_run_probe_ok": False,
            "mpp_run_probe_error": "result mismatch",
            "mpp_run_probe_max_abs_error": 31.0,
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        run_mpp_probe=True,
    )

    assert capability.mpp_compile_probe_requested is True
    assert capability.mpp_run_probe_requested is True
    assert capability.mpp_run_probe_ran is True
    assert capability.mpp_run_probe_ok is False
    assert capability.mpp_runtime_available is False
    assert capability.recommended_backend == "mpsgraph_prefill_fallback"
    assert "runtime MPP matmul2d run probe failed" in capability.reasons
    assert capability.prefill_neural_accelerator_status["status"] == (
        "run_probe_failed"
    )


def test_prefill_backend_compacts_long_mpp_compile_error(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)
    long_error = "metal_stdlib:\n" + ("use of undeclared identifier mpp\n" * 20)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "supports_metal4_family": True,
            "responds_new_mtl4_command_queue": True,
            "responds_new_tensor": True,
            "responds_tensor_size_align": True,
            "responds_new_compiler": True,
            "can_allocate_tiny_ml_tensor": True,
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": False,
            "mpp_compile_error": long_error,
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        compile_mpp_probe=True,
    )

    assert capability.mpp_compile_probe_ran is True
    assert capability.mpp_compile_probe_ok is False
    assert capability.mpp_compile_error is not None
    assert len(capability.mpp_compile_error) <= 240
    assert capability.mpp_compile_error.startswith("metal_stdlib:")
    assert capability.mpp_compile_error.endswith("...")


def test_prefill_backend_mpp_runtime_requires_sdk_symbols(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=False)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "supports_metal4_family": True,
            "responds_new_mtl4_command_queue": True,
            "responds_new_tensor": True,
            "responds_tensor_size_align": True,
            "responds_new_compiler": True,
            "can_allocate_tiny_ml_tensor": True,
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": True,
            "mpp_compile_variant": "metal_mpp",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        compile_mpp_probe=True,
    )

    assert not capability.mpp_tensor_ops_symbol_declared
    assert not capability.mpp_runtime_available
    assert capability.recommended_backend == "mpsgraph_prefill_fallback"
    assert capability.prefill_neural_accelerator_status["status"] == (
        "missing_public_mpp_symbols"
    )


def test_prefill_backend_mpp_status_prefers_missing_symbols_over_run_failure(
    tmp_path: Path,
) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=False)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "supports_metal4_family": True,
            "responds_new_mtl4_command_queue": True,
            "responds_new_tensor": True,
            "responds_tensor_size_align": True,
            "responds_new_compiler": True,
            "can_allocate_tiny_ml_tensor": True,
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": False,
            "mpp_compile_error": "use of undeclared identifier mpp",
            "mpp_run_probe_ran": True,
            "mpp_run_probe_ok": False,
            "mpp_run_probe_error": "metal_mpp file not found",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        run_mpp_probe=True,
    )

    status = capability.prefill_neural_accelerator_status
    assert status["status"] == "missing_public_mpp_symbols"
    assert status["reason"] == (
        "Metal 4 ML runtime is visible but public MPP symbols are missing"
    )
    assert status["mpp_run_probe_ok"] is False
    assert status["mpp_run_probe_error"] == "metal_mpp file not found"


def test_prefill_backend_recommendation_requires_full_metal4_runtime(
    tmp_path: Path,
) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "supports_metal4_family": True,
            "responds_new_mtl4_command_queue": True,
            "responds_new_tensor": False,
            "responds_tensor_size_align": False,
            "responds_new_compiler": True,
            "can_allocate_tiny_ml_tensor": False,
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": True,
            "mpp_compile_variant": "metal_mpp",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        compile_mpp_probe=True,
    )

    assert not capability.metal4_ml_runtime_available
    assert not capability.mpp_runtime_available
    assert capability.recommended_backend == "mpsgraph_prefill_fallback"
    assert "default Metal device lacks newTensorWithDescriptor:error:" in (
        capability.reasons
    )
    assert "default Metal device lacks tensorSizeAndAlignWithDescriptor:" in (
        capability.reasons
    )


def test_prefill_backend_recommendation_requires_tiny_ml_tensor_allocation(
    tmp_path: Path,
) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": True,
            "supports_metal4_family": True,
            "responds_new_mtl4_command_queue": True,
            "responds_new_tensor": True,
            "responds_tensor_size_align": True,
            "responds_new_compiler": True,
            "can_allocate_tiny_ml_tensor": False,
            "tensor_error": "MTL allocation denied",
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": True,
            "mpp_compile_variant": "metal_mpp",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        compile_mpp_probe=True,
    )

    assert not capability.metal4_ml_runtime_available
    assert not capability.mpp_runtime_available
    assert capability.recommended_backend == "mpsgraph_prefill_fallback"
    assert capability.tensor_error == "MTL allocation denied"
    assert "default Metal device cannot allocate a tiny ML tensor" in (
        capability.reasons
    )
    assert "tiny ML tensor allocation failed: MTL allocation denied" in (
        capability.reasons
    )


def test_prefill_backend_reports_no_default_metal_device(
    tmp_path: Path,
) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)
    probe = _write_probe(
        tmp_path / "probe.py",
        {
            "ok": False,
            "device_name": "",
            "supports_metal4_family": False,
            "responds_new_mtl4_command_queue": False,
            "responds_new_tensor": False,
            "responds_tensor_size_align": False,
            "responds_new_compiler": False,
            "can_allocate_tiny_ml_tensor": False,
            "tensor_error": "no Metal device",
            "mpp_compile_probe_ran": True,
            "mpp_compile_probe_ok": False,
            "mpp_compile_error": "no Metal device",
        },
    )

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=probe,
        compile_mpp_probe=True,
    )

    assert not capability.metal4_ml_runtime_available
    assert not capability.mps_graph_runtime_available
    assert capability.device_name is None
    assert capability.tensor_error == "no Metal device"
    assert capability.recommended_backend == "custom_metal_prefill_fallback"
    assert "host probe could not create a default Metal device" in capability.reasons
    assert "tiny ML tensor allocation failed: no Metal device" in capability.reasons
    assert "default Metal device lacks newTensorWithDescriptor:error:" not in (
        capability.reasons
    )


def test_prefill_backend_rejects_missing_sdk(tmp_path: Path) -> None:
    with pytest.raises(PrefillBackendError, match="SDK path"):
        inspect_prefill_backend(sdk_path=tmp_path / "missing", run_host_probe=False)


def test_prefill_backend_reports_host_probe_error(tmp_path: Path) -> None:
    sdk = _fake_sdk(tmp_path)
    missing_probe = tmp_path / "missing-probe"

    capability = inspect_prefill_backend(
        sdk_path=sdk,
        probe_binary=missing_probe,
        run_host_probe=True,
    )

    assert capability.host_probe_requested is True
    assert capability.host_probe_path == missing_probe
    assert capability.host_probe_ran is False
    assert capability.host_probe_error == "probe binary does not exist"
    assert any("probe binary does not exist" in item for item in capability.reasons)


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("nan"), True))
def test_prefill_backend_rejects_invalid_probe_timeout(
    tmp_path: Path,
    timeout: object,
) -> None:
    sdk = _fake_sdk(tmp_path)

    with pytest.raises(PrefillBackendError, match="probe_timeout_seconds"):
        inspect_prefill_backend(
            sdk_path=sdk,
            run_host_probe=False,
            probe_timeout_seconds=timeout,
        )


def test_prefill_backend_cli_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sdk = _fake_sdk(tmp_path)
    report_path = tmp_path / "reports" / "prefill-backend.json"

    status = cli_main(
        [
            "prefill-backend",
            "--sdk-path",
            str(sdk),
            "--no-host-probe",
            "--write-report",
            str(report_path),
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload == payload
    assert not report_path.with_name(f"{report_path.name}.tmp").exists()
    assert payload["recommended_backend"] == "mpsgraph_prefill_fallback"
    assert payload["host_probe_requested"] is False
    assert payload["host_probe_path"] is None
    assert payload["host_probe_sha256"] is None
    assert payload["host_probe_error"] is None
    assert payload["host_probe_ran"] is False
    assert payload["mpp_compile_probe_requested"] is False
    assert payload["mps_graph_probe_requested"] is False
    assert payload["mps_graph_probe_ran"] is False
    assert payload["mps_graph_probe_ok"] is None
    assert payload["mps_graph_probe_error"] is None
    assert payload["responds_tensor_size_align"] is None
    assert payload["tensor_error"] is None
    assert payload["mps_graph_runtime_available"] is True
    assert payload["metal4_ml_runtime_available"] is False
    assert payload["mpp_runtime_available"] is False
    assert payload["prefill_acceleration_runtimes"] == ["mpsgraph-f32"]
    assert payload["selectable_accelerated_prefill_backends"] == ["mpsgraph-f32"]
    assert payload["selectable_prefill_acceleration_available"] is True
    assert payload["validated_accelerated_prefill_backends"] == []
    assert payload["validated_prefill_acceleration_available"] is False
    assert payload["prefill_acceleration_runtime_gaps"] == []
    assert payload["prefill_neural_accelerator_status"] == {
        "runtime": "mpp_tensor_ops_prefill",
        "execution_path": "mpp_tensor_ops_gpu_neural_accelerator",
        "status": "unavailable",
        "ready_for_generation": False,
        "runtime_visible": False,
        "selectable": False,
        "metal4_ml_runtime_available": False,
        "mpp_tensor_ops_symbol_declared": False,
        "mpp_compile_probe_requested": False,
        "mpp_compile_probe_ran": False,
        "mpp_compile_probe_ok": None,
        "mpp_run_probe_requested": False,
        "mpp_run_probe_ran": False,
        "mpp_run_probe_ok": None,
        "mpp_run_probe_error": None,
        "mpp_run_probe_max_abs_error": None,
        "mpp_run_probe_kernel_variant": None,
        "mpp_run_probe_shape": None,
        "mpp_run_probe_dtype": None,
        "mpp_run_probe_execution_path": None,
        "recommended_backend": "mpsgraph_prefill_fallback",
        "reason": "MPP tensor ops prefill runtime is not available",
    }
    assert payload["suggested_prefill_acceleration_flags"] == {
        "source": "prefill_backend_probe",
        "prefill_linear_backend": "mpsgraph-f32",
        "prefill_acceleration_runtimes": ["mpsgraph-f32"],
        "selectable_accelerated_prefill_backends": ["mpsgraph-f32"],
        "validated_accelerated_prefill_backends": [],
        "prefill_acceleration_runtime_gaps": [],
        "prefill_neural_accelerator_status": payload[
            "prefill_neural_accelerator_status"
        ],
        "runtime_probe_backend": "mpsgraph-f32",
        "runtime_probe_required": True,
        "runtime_probe_satisfied": False,
        "runtime_probe_argv": ["--run-mpsgraph-probe"],
        "argv": [
            "--prefill-linear-backend",
            "mpsgraph-f32",
            "--require-prefill-acceleration",
            "--run-mpsgraph-probe",
        ],
    }


def test_prefill_backend_cli_runs_mpsgraph_probe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sdk = _fake_sdk(tmp_path)
    probe = _write_arg_sensitive_probe(tmp_path / "probe.py")

    status = cli_main(
        [
            "prefill-backend",
            "--sdk-path",
            str(sdk),
            "--probe-binary",
            str(probe),
            "--run-mpsgraph-probe",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mps_graph_probe_requested"] is True
    assert payload["mps_graph_probe_ran"] is True
    assert payload["mps_graph_probe_ok"] is True
    assert payload["mps_graph_runtime_available"] is True
    assert payload["validated_accelerated_prefill_backends"] == ["mpsgraph-f32"]
    assert payload["validated_prefill_acceleration_available"] is True


def test_prefill_backend_cli_runs_mpp_probe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sdk = _fake_sdk(tmp_path, include_mpp_symbol=True)
    probe = _write_arg_sensitive_probe(tmp_path / "probe.py")

    status = cli_main(
        [
            "prefill-backend",
            "--sdk-path",
            str(sdk),
            "--probe-binary",
            str(probe),
            "--run-mpp-probe",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mpp_compile_probe_requested"] is True
    assert payload["mpp_run_probe_requested"] is True
    assert payload["mpp_run_probe_ran"] is True
    assert payload["mpp_run_probe_ok"] is True
    assert payload["mpp_run_probe_max_abs_error"] == 0.0
    assert payload["mpp_run_probe_kernel_variant"] == "fake_mpp"
    assert payload["mpp_run_probe_shape"] == "32x32x32"
    assert payload["mpp_run_probe_dtype"] == "half"
    assert payload["mpp_run_probe_execution_path"] == "mpp::tensor_ops::matmul2d"
    assert payload["prefill_neural_accelerator_status"]["status"] == (
        "metal4_ml_runtime_unavailable"
    )
    assert payload["prefill_neural_accelerator_status"][
        "mpp_run_probe_shape"
    ] == "32x32x32"


def test_prefill_plan_can_attach_backend_capability(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sdk = _fake_sdk(tmp_path)

    status = cli_main(
        [
            "prefill-plan",
            str(FIXTURES / "glm_moe_dsa_config.json"),
            "--prompt-tokens",
            "128",
            "--inspect-backend",
            "--sdk-path",
            str(sdk),
            "--no-host-probe",
            "--json",
        ]
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mpp_candidate_ops"] == 11
    assert payload["effective_metal4_candidate_ops"] == 0
    assert payload["effective_mpp_candidate_ops"] == 0
    assert payload["backend_capability"]["metal4_machine_learning_declared"] is True
    assert payload["backend_capability"]["metal4_ml_runtime_available"] is False
    assert payload["backend_capability"]["mpp_runtime_available"] is False
    assert payload["backend_capability"]["prefill_acceleration_runtimes"] == [
        "mpsgraph-f32"
    ]
    assert payload["backend_capability"][
        "selectable_accelerated_prefill_backends"
    ] == ["mpsgraph-f32"]
    assert payload["backend_capability"]["prefill_neural_accelerator_status"][
        "status"
    ] == "unavailable"


def test_prefill_acceleration_helpers_keep_mpp_candidate_unselectable() -> None:
    capability = type(
        "Capability",
        (),
        {
            "mps_graph_runtime_available": False,
            "mpp_runtime_available": True,
        },
    )()

    assert prefill_acceleration_runtimes(capability) == ("mpp_tensor_ops_prefill",)
    assert selectable_accelerated_prefill_backends(capability) == ()
    assert validated_accelerated_prefill_backends(capability) == ()
    assert prefill_acceleration_runtime_gaps(capability) == (
        {
            "runtime": "mpp_tensor_ops_prefill",
            "reason": (
                "MPP tensor ops runtime is visible but no selectable MPP "
                "prefill execution backend is implemented"
            ),
        },
    )
    assert prefill_neural_accelerator_status(capability)["status"] == (
        "runtime_visible_not_selectable"
    )
    assert suggested_prefill_acceleration_flags(capability) is None
