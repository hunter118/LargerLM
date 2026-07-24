from __future__ import annotations

import platform
from pathlib import Path

import pytest

from largerlm.colibri_m5 import (
    ColibriM5Error,
    LaunchConfig,
    colibri_command,
    colibri_environment,
    parse_memory_pressure_free_percent,
    parse_vm_stat_available_bytes,
    run_guarded,
    validate_launch,
)


def _config(tmp_path: Path, **overrides: object) -> LaunchConfig:
    engine_dir = tmp_path / "c"
    engine_dir.mkdir()
    engine = engine_dir / "coli"
    engine.write_text("#!/bin/sh\n", encoding="ascii")
    engine.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="ascii")
    usage = model / ".coli_usage"
    usage.write_text("3 4 5\n", encoding="ascii")
    values: dict[str, object] = {
        "engine": engine,
        "model": model,
        "prompt": "hello",
        "usage_profile": usage,
    }
    values.update(overrides)
    return LaunchConfig(**values)  # type: ignore[arg-type]


def test_parse_vm_stat_available_bytes_counts_reclaimable_pages() -> None:
    output = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 10.
Pages inactive: 20.
Pages speculative: 3.
Pages purgeable: 2.
"""
    assert parse_vm_stat_available_bytes(output) == 35 * 16384


def test_parse_memory_pressure_free_percent() -> None:
    output = "System-wide memory free percentage: 27%\n"
    assert parse_memory_pressure_free_percent(output) == 27


def test_colibri_m5_environment_enables_measured_fast_path(tmp_path: Path) -> None:
    config = _config(tmp_path, profile=True)
    env = colibri_environment(
        config,
        {
            "PATH": "/bin",
            "CACHE_ROUTE": "1",
            "COLI_METAL_CACHE_ROUTE": "1",
        },
    )

    assert env["COLI_METAL"] == "1"
    assert env["COLI_METAL_RESSET"] == "1"
    assert env["CACHE_ROUTE"] == "0"
    assert env["COLI_METAL_CACHE_ROUTE"] == "0"
    assert env["COLI_KV_SLOTS"] == "1"
    assert env["COLI_MAX_QUEUE"] == "1"
    assert env["CTX"] == "4096"
    assert env["DIRECT"] == "1"
    assert env["PIPE_WORKERS"] == "8"
    assert env["MTP"] == "0"
    assert env["PILOT"] == "0"
    assert env["PIN_GB"] == "46"
    assert env["PROF"] == "1"
    assert env["PATH"] == "/bin"


def test_experimental_fast_mode_enables_measured_gpu_cache_route(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, mode="experimental-fast")
    env = colibri_environment(
        config,
        {
            "CACHE_ROUTE": "stale",
            "ROUTE_J": "stale",
            "ROUTE_M": "stale",
        },
    )

    assert env["CACHE_ROUTE"] == "1"
    assert env["COLI_METAL_CACHE_ROUTE"] == "1"
    assert env["ROUTE_J"] == "2"
    assert env["ROUTE_M"] == "32"
    assert env["ROUTE_P"] == "0"
    assert env["ROUTE_ALPHA"] == "1"


def test_colibri_command_is_quality_preserving_greedy_profile(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, ngen=256)

    assert colibri_command(config) == [
        str(config.engine),
        "run",
        "hello",
        "--model",
        str(config.model),
        "--ram",
        "110",
        "--ctx",
        "4096",
        "--ngen",
        "256",
        "--temp",
        "0.0",
    ]


def test_colibri_web_command_uses_single_slot_loopback_server(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        prompt="",
        interface="web",
        port=8123,
        ngen=512,
        context_tokens=32768,
    )
    web_dist = config.engine.parent.parent / "web/dist"
    web_dist.mkdir(parents=True)
    (web_dist / "index.html").write_text("<html></html>", encoding="ascii")

    assert colibri_command(config) == [
        str(config.engine),
        "web",
        "--model",
        str(config.model),
        "--ram",
        "110",
        "--ctx",
        "32768",
        "--ngen",
        "512",
        "--temp",
        "0.0",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--kv-slots",
        "1",
        "--max-queue",
        "1",
        "--no-browser",
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"ram_gib": 111}, "ram_gib"),
        ({"pin_gib": 47}, "pin_gib"),
        ({"ram_gib": 40, "pin_gib": 40}, "smaller"),
        ({"ngen": 0}, "ngen"),
        ({"context_tokens": 32769}, "context_tokens"),
        ({"ngen": 4097, "context_tokens": 4096}, "context_tokens"),
        ({"prompt": "  "}, "prompt"),
        ({"mode": "turbo"}, "mode"),
        ({"interface": "desktop"}, "interface"),
        (
            {"prompt": "", "interface": "web", "host": "0.0.0.0"},
            "loopback",
        ),
    ),
)
def test_validate_launch_rejects_unsafe_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr("largerlm.colibri_m5.sys.platform", "darwin")

    with pytest.raises(ColibriM5Error, match=message):
        validate_launch(_config(tmp_path, **overrides))


def test_guard_restores_usage_when_process_launch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr("largerlm.colibri_m5.sys.platform", "darwin")
    monkeypatch.setattr(
        "largerlm.colibri_m5.available_memory_bytes",
        lambda: 100 * 1024**3,
    )

    def fail_launch(*args: object, **kwargs: object) -> None:
        raise OSError("launch failed")

    monkeypatch.setattr("largerlm.colibri_m5.subprocess.Popen", fail_launch)
    config = _config(tmp_path)
    original = config.usage_profile.read_bytes()

    with pytest.raises(OSError, match="launch failed"):
        run_guarded(
            config,
            preserve_usage=True,
            reset_runtime_usage=True,
        )

    assert config.usage_profile.read_bytes() == original
