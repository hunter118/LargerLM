from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


GIB = 1024**3
M5_MAX_RAM_GIB = 110
M5_MAX_PIN_GIB = 46
M5_MAX_RSS_GIB = 105
M5_MIN_START_AVAILABLE_GIB = 24
M5_MIN_PRESSURE_FREE_PERCENT = 10
QUALITY_MODE = "quality"
EXPERIMENTAL_FAST_MODE = "experimental-fast"
LAUNCH_MODES = (QUALITY_MODE, EXPERIMENTAL_FAST_MODE)
RUN_INTERFACE = "run"
WEB_INTERFACE = "web"
LAUNCH_INTERFACES = (RUN_INTERFACE, WEB_INTERFACE)
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class ColibriM5Error(RuntimeError):
    """Raised when the guarded Colibri launch is unsafe or fails."""


@dataclass(frozen=True)
class MemorySample:
    available_bytes: int
    process_rss_bytes: int
    pressure_free_percent: int


@dataclass(frozen=True)
class LaunchConfig:
    engine: Path
    model: Path
    prompt: str
    usage_profile: Path
    ram_gib: int = M5_MAX_RAM_GIB
    pin_gib: int = M5_MAX_PIN_GIB
    ngen: int = 64
    temperature: float = 0.0
    profile: bool = False
    mode: str = QUALITY_MODE
    interface: str = RUN_INTERFACE
    host: str = "127.0.0.1"
    port: int = 8000


def parse_vm_stat_available_bytes(output: str) -> int:
    match = re.search(r"page size of\s+(\d+)\s+bytes", output)
    if match is None:
        raise ColibriM5Error("vm_stat output is missing its page size")
    page_size = int(match.group(1))
    pages: dict[str, int] = {}
    for line in output.splitlines():
        field = re.match(r"([^:]+):\s+(\d+)\.", line)
        if field is not None:
            pages[field.group(1).strip()] = int(field.group(2))
    required = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    missing = [name for name in required if name not in pages]
    if missing:
        raise ColibriM5Error(
            "vm_stat output is missing fields: " + ", ".join(missing)
        )
    return sum(pages[name] for name in required) * page_size


def available_memory_bytes() -> int:
    result = subprocess.run(
        ["/usr/bin/vm_stat"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_vm_stat_available_bytes(result.stdout)


def parse_memory_pressure_free_percent(output: str) -> int:
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    if match is None:
        raise ColibriM5Error(
            "memory_pressure output is missing the system-wide free percentage"
        )
    return int(match.group(1))


def memory_pressure_free_percent() -> int:
    result = subprocess.run(
        ["/usr/bin/memory_pressure"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_memory_pressure_free_percent(result.stdout)


def process_group_rss_bytes(process_group: int) -> int:
    result = subprocess.run(
        ["/bin/ps", "-axo", "pgid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    total_kib = 0
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and int(fields[0]) == process_group:
            total_kib += int(fields[1])
    return total_kib * 1024


def validate_launch(config: LaunchConfig) -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ColibriM5Error("the guarded profile requires Apple Silicon macOS")
    if not config.engine.is_file():
        raise ColibriM5Error(f"Colibri CLI does not exist: {config.engine}")
    if not os.access(config.engine, os.X_OK):
        raise ColibriM5Error(f"Colibri CLI is not executable: {config.engine}")
    if not config.model.is_dir():
        raise ColibriM5Error(f"model directory does not exist: {config.model}")
    if not (config.model / "config.json").is_file():
        raise ColibriM5Error(f"model config is missing: {config.model / 'config.json'}")
    if not config.usage_profile.is_file():
        raise ColibriM5Error(
            f"fixed Colibri usage profile does not exist: {config.usage_profile}"
        )
    if not 1 <= config.ram_gib <= M5_MAX_RAM_GIB:
        raise ColibriM5Error(
            f"ram_gib must be in [1, {M5_MAX_RAM_GIB}] for the safe M5 profile"
        )
    if not 0 <= config.pin_gib <= M5_MAX_PIN_GIB:
        raise ColibriM5Error(
            f"pin_gib must be in [0, {M5_MAX_PIN_GIB}] for the safe M5 profile"
        )
    if config.pin_gib >= config.ram_gib:
        raise ColibriM5Error("pin_gib must be smaller than ram_gib")
    if config.ngen <= 0:
        raise ColibriM5Error("ngen must be positive")
    if config.interface not in LAUNCH_INTERFACES:
        raise ColibriM5Error(
            f"interface must be one of: {', '.join(LAUNCH_INTERFACES)}"
        )
    if config.interface == RUN_INTERFACE and not config.prompt.strip():
        raise ColibriM5Error("prompt must not be empty")
    if config.interface == WEB_INTERFACE:
        if config.host not in LOOPBACK_HOSTS:
            raise ColibriM5Error(
                "the guarded web profile only binds to a loopback address"
            )
        if not 1 <= config.port <= 65535:
            raise ColibriM5Error("port must be in [1, 65535]")
        web_index = config.engine.parent.parent / "web/dist/index.html"
        if not web_index.is_file():
            raise ColibriM5Error(
                f"Colibri web UI is not built: {web_index}"
            )
    if config.mode not in LAUNCH_MODES:
        raise ColibriM5Error(
            f"mode must be one of: {', '.join(LAUNCH_MODES)}"
        )


def colibri_environment(
    config: LaunchConfig,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update(
        {
            "AUTOPIN": "0",
            "CACHE_ROUTE": "0",
            "COLI_METAL": "1",
            "COLI_METAL_CACHE_ROUTE": "0",
            "COLI_METAL_RESSET": "1",
            "COLI_NO_OMP_TUNE": "1",
            "DIRECT": "1",
            "KVSAVE": "0",
            "COLI_KV_SLOTS": "1",
            "COLI_MAX_QUEUE": "1",
            "MTP": "0",
            "PILOT": "0",
            "PILOT_REAL": "0",
            "PILOT_TWO": "0",
            "PIN": str(config.usage_profile),
            "PIN_GB": str(config.pin_gib),
            "PIPE": "1",
            "PIPE_WORKERS": "8",
            "PROF": "1" if config.profile else "0",
            "TEMP": str(config.temperature),
        }
    )
    if config.mode == EXPERIMENTAL_FAST_MODE:
        env.update(
            {
                "CACHE_ROUTE": "1",
                "COLI_METAL_CACHE_ROUTE": "1",
                "ROUTE_ALPHA": "1",
                "ROUTE_J": "2",
                "ROUTE_M": "32",
                "ROUTE_P": "0",
            }
        )
    return env


def colibri_command(config: LaunchConfig) -> list[str]:
    if config.interface == WEB_INTERFACE:
        return [
            str(config.engine),
            "web",
            "--model",
            str(config.model),
            "--ram",
            str(config.ram_gib),
            "--ngen",
            str(config.ngen),
            "--temp",
            str(config.temperature),
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--kv-slots",
            "1",
            "--max-queue",
            "1",
            "--no-browser",
        ]
    return [
        str(config.engine),
        "run",
        config.prompt,
        "--model",
        str(config.model),
        "--ram",
        str(config.ram_gib),
        "--ngen",
        str(config.ngen),
        "--temp",
        str(config.temperature),
    ]


def _stop_process_group(process: subprocess.Popen[bytes], reason: str) -> None:
    print(f"\n[LargerLM guard] stopping Colibri: {reason}", file=sys.stderr)
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_guarded(
    config: LaunchConfig,
    *,
    max_rss_gib: int = M5_MAX_RSS_GIB,
    min_start_available_gib: int = M5_MIN_START_AVAILABLE_GIB,
    min_pressure_free_percent: int = M5_MIN_PRESSURE_FREE_PERCENT,
    sample_seconds: float = 2.0,
    preserve_usage: bool = False,
    reset_runtime_usage: bool = False,
    capture_usage: Path | None = None,
) -> int:
    validate_launch(config)
    if not 1 <= max_rss_gib <= M5_MAX_RAM_GIB:
        raise ColibriM5Error(
            f"max_rss_gib must be in [1, {M5_MAX_RAM_GIB}]"
        )
    if not 1 <= min_pressure_free_percent <= 50:
        raise ColibriM5Error(
            "min_pressure_free_percent must be in [1, 50]"
        )
    if sample_seconds < 0.25:
        raise ColibriM5Error("sample_seconds must be at least 0.25")
    if reset_runtime_usage and not preserve_usage:
        raise ColibriM5Error(
            "reset_runtime_usage requires preserve_usage so history cannot be lost"
        )

    start_available = available_memory_bytes()
    if start_available < min_start_available_gib * GIB:
        raise ColibriM5Error(
            "refusing launch: reclaimable memory is "
            f"{start_available / GIB:.1f} GiB, below the "
            f"{min_start_available_gib} GiB startup reserve"
        )

    usage_path = config.model / ".coli_usage"
    backup_dir: tempfile.TemporaryDirectory[str] | None = None
    backup_path: Path | None = None
    usage_existed = usage_path.exists()
    if preserve_usage and usage_existed:
        backup_dir = tempfile.TemporaryDirectory(prefix="largerlm-colibri-usage-")
        backup_path = Path(backup_dir.name) / ".coli_usage"
        shutil.copy2(usage_path, backup_path)
    if reset_runtime_usage and usage_path.exists():
        usage_path.unlink()

    process: subprocess.Popen[bytes] | None = None
    peak_rss = 0
    minimum_pressure_free = 100
    guard_reason: str | None = None
    stop_signal = 0

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_signal
        stop_signal = signum

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        print(
            "[LargerLM guard] "
            f"start available={start_available / GIB:.1f} GiB, "
            f"RAM cap={config.ram_gib} GiB, pin={config.pin_gib} GiB, "
            f"RSS stop={max_rss_gib} GiB, "
            f"pressure-free stop={min_pressure_free_percent}%",
            file=sys.stderr,
        )
        if config.mode == EXPERIMENTAL_FAST_MODE:
            print(
                "[LargerLM guard] EXPERIMENTAL: cache-aware routing changes "
                "expert choices and is not output-equivalent to quality mode",
                file=sys.stderr,
            )
        process = subprocess.Popen(
            colibri_command(config),
            env=colibri_environment(config),
            start_new_session=True,
        )
        while process.poll() is None:
            time.sleep(sample_seconds)
            if stop_signal:
                guard_reason = f"received signal {stop_signal}"
                _stop_process_group(process, guard_reason)
                break
            rss = process_group_rss_bytes(process.pid)
            pressure_free = memory_pressure_free_percent()
            peak_rss = max(peak_rss, rss)
            minimum_pressure_free = min(minimum_pressure_free, pressure_free)
            if rss > max_rss_gib * GIB:
                guard_reason = (
                    f"process-group RSS {rss / GIB:.1f} GiB exceeded "
                    f"{max_rss_gib} GiB"
                )
                _stop_process_group(process, guard_reason)
                break
            if pressure_free < min_pressure_free_percent:
                guard_reason = (
                    f"memory_pressure free percentage {pressure_free}% fell below "
                    f"{min_pressure_free_percent}%"
                )
                _stop_process_group(process, guard_reason)
                break
        return_code = process.wait()
    finally:
        if capture_usage is not None and usage_path.exists():
            capture_usage.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(usage_path, capture_usage)
        if preserve_usage:
            if backup_path is not None:
                shutil.copy2(backup_path, usage_path)
            elif not usage_existed and usage_path.exists():
                usage_path.unlink()
        if backup_dir is not None:
            backup_dir.cleanup()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    print(
        "[LargerLM guard] "
        f"peak RSS={peak_rss / GIB:.1f} GiB, "
        f"minimum pressure-free={minimum_pressure_free}%, "
        f"exit={return_code}",
        file=sys.stderr,
    )
    if guard_reason is not None and not stop_signal:
        raise ColibriM5Error(guard_reason)
    return 0 if stop_signal else return_code


def _guard_process_command(pid: int) -> str:
    result = subprocess.run(
        ["/bin/ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def start_detached_web(
    argv: Sequence[str],
    *,
    pid_file: Path,
    log_file: Path,
) -> int:
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            old_pid = 0
        if old_pid > 0 and _guard_process_command(old_pid):
            raise ColibriM5Error(
                f"guarded Colibri web process is already running: PID {old_pid}"
            )
        pid_file.unlink(missing_ok=True)

    child_argv = [argument for argument in argv if argument != "--detach"]
    launcher = Path(sys.argv[0]).expanduser().resolve()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, str(launcher), *child_argv],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    pid_file.write_text(f"{process.pid}\n", encoding="ascii")
    print(
        f"guarded Colibri web monitor started: PID {process.pid}\n"
        f"log: {log_file}"
    )
    return 0


def stop_detached_web(pid_file: Path, timeout: float = 20.0) -> int:
    if not pid_file.is_file():
        raise ColibriM5Error(f"web PID file does not exist: {pid_file}")
    try:
        pid = int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, ValueError) as exc:
        raise ColibriM5Error(f"invalid web PID file: {pid_file}") from exc
    command = _guard_process_command(pid)
    if "run_colibri_m5.py" not in command or "--web" not in command:
        raise ColibriM5Error(
            f"refusing to signal PID {pid}; it is not a guarded Colibri web monitor"
        )
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _guard_process_command(pid):
            pid_file.unlink(missing_ok=True)
            print(f"guarded Colibri web monitor stopped: PID {pid}")
            return 0
        time.sleep(0.25)
    raise ColibriM5Error(
        f"web monitor PID {pid} did not stop within {timeout:.0f}s"
    )


def _first_existing(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    configured_engine = os.environ.get("COLIBRI_CLI")
    engine_candidates = [
        project_root / "third_party/colibri/c/coli",
        project_root / "third_party/colibri/c/colibri",
        Path("/private/tmp/colibri-run-81f08a/c/coli"),
        Path("/private/tmp/colibri-run-81f08a/c/colibri"),
    ]
    if configured_engine:
        engine_candidates.insert(0, Path(configured_engine))
    default_engine = _first_existing(
        engine_candidates
    )
    default_model = project_root / "artifacts/colibri-glm5.2-int4"
    default_usage_profile = (
        project_root / "profiles/glm-5.2-colibri-usage-259200.txt"
    )
    runtime_dir = project_root / "runtime"
    parser = argparse.ArgumentParser(
        description="Run Colibri's GLM-5.2 Metal path with M5 Max 128 GiB guards."
    )
    parser.add_argument("prompt", nargs="?", default="")
    parser.add_argument("--engine", type=Path, default=default_engine)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument(
        "--usage-profile",
        type=Path,
        default=default_usage_profile,
    )
    parser.add_argument("--ngen", type=int, default=64)
    parser.add_argument("--ram-gib", type=int, default=M5_MAX_RAM_GIB)
    parser.add_argument("--pin-gib", type=int, default=M5_MAX_PIN_GIB)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        choices=LAUNCH_MODES,
        default=QUALITY_MODE,
        help=(
            "quality preserves the model router; experimental-fast trades "
            "routing fidelity for roughly 5 tok/s"
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--preserve-usage", action="store_true")
    parser.add_argument("--reset-runtime-usage", action="store_true")
    parser.add_argument("--capture-usage", type=Path)
    parser.add_argument("--max-rss-gib", type=int, default=M5_MAX_RSS_GIB)
    parser.add_argument(
        "--min-start-available-gib",
        type=int,
        default=M5_MIN_START_AVAILABLE_GIB,
    )
    parser.add_argument(
        "--min-pressure-free-percent",
        type=int,
        default=M5_MIN_PRESSURE_FREE_PERCENT,
    )
    parser.add_argument("--sample-seconds", type=float, default=2.0)
    parser.add_argument(
        "--web",
        action="store_true",
        help="serve Colibri's browser chat UI instead of one prompt",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="run the guarded web monitor in the background",
    )
    parser.add_argument(
        "--stop-web",
        action="store_true",
        help="stop the detached guarded web monitor and its model process",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=runtime_dir / "colibri-web.pid",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=runtime_dir / "colibri-web.log",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    pid_file = args.pid_file.expanduser().resolve()
    log_file = args.log_file.expanduser().resolve()
    if args.stop_web:
        try:
            return stop_detached_web(pid_file)
        except (ColibriM5Error, OSError, subprocess.SubprocessError) as exc:
            print(f"run_colibri_m5: {exc}", file=sys.stderr)
            return 2
    usage_profile = args.usage_profile or args.model / ".coli_usage"
    config = LaunchConfig(
        engine=args.engine.expanduser().resolve(),
        model=args.model.expanduser().resolve(),
        prompt=args.prompt,
        usage_profile=usage_profile.expanduser().resolve(),
        ram_gib=args.ram_gib,
        pin_gib=args.pin_gib,
        ngen=args.ngen,
        temperature=args.temperature,
        profile=args.profile,
        mode=args.mode,
        interface=WEB_INTERFACE if args.web else RUN_INTERFACE,
        host=args.host,
        port=args.port,
    )
    try:
        validate_launch(config)
        if args.detach:
            if config.interface != WEB_INTERFACE:
                raise ColibriM5Error("--detach requires --web")
            return start_detached_web(
                raw_argv,
                pid_file=pid_file,
                log_file=log_file,
            )
        if args.dry_run:
            print(" ".join(colibri_command(config)))
            for key, value in sorted(colibri_environment(config, {}).items()):
                print(f"{key}={value}")
            return 0
        try:
            return run_guarded(
                config,
                max_rss_gib=args.max_rss_gib,
                min_start_available_gib=args.min_start_available_gib,
                min_pressure_free_percent=args.min_pressure_free_percent,
                sample_seconds=args.sample_seconds,
                preserve_usage=args.preserve_usage,
                reset_runtime_usage=args.reset_runtime_usage,
                capture_usage=(
                    args.capture_usage.expanduser().resolve()
                    if args.capture_usage is not None
                    else None
                ),
            )
        finally:
            if config.interface == WEB_INTERFACE:
                try:
                    if (
                        pid_file.is_file()
                        and int(pid_file.read_text(encoding="ascii").strip())
                        == os.getpid()
                    ):
                        pid_file.unlink()
                except (OSError, ValueError):
                    pass
    except (ColibriM5Error, OSError, subprocess.SubprocessError) as exc:
        print(f"run_colibri_m5: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
