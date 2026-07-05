from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareInfo:
    chip_name: str
    unified_memory_bytes: int | None
    gpu_cores: int | None
    apple_silicon_generation: int | None = None
    apple_silicon_tier: str | None = None

    @property
    def unified_memory_gib(self) -> float | None:
        if self.unified_memory_bytes is None:
            return None
        return self.unified_memory_bytes / (1024**3)


def parse_apple_silicon_chip(chip_name: str | None) -> tuple[int | None, str | None]:
    if not chip_name:
        return None, None
    match = re.search(
        r"\b(?:Apple\s+)?M(?P<generation>\d+)(?:\s+(?P<tier>Ultra|Max|Pro))?\b",
        chip_name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, None
    generation = int(match.group("generation"))
    raw_tier = match.group("tier")
    tier = raw_tier.capitalize() if raw_tier else None
    return generation, tier


def _run(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
    except Exception:
        return None
    return result.stdout.strip()


def detect_hardware() -> HardwareInfo:
    chip = _run(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
    memory_bytes: int | None = None
    memory = _run(["/usr/sbin/sysctl", "-n", "hw.memsize"])
    if memory:
        try:
            memory_bytes = int(memory)
        except ValueError:
            memory_bytes = None

    gpu_cores = None
    hardware = _run(["system_profiler", "SPHardwareDataType"])
    if hardware:
        chip_match = re.search(r"^\s*Chip:\s*(.+)$", hardware, re.MULTILINE)
        if chip_match and not chip:
            chip = chip_match.group(1).strip()
        mem_match = re.search(r"^\s*Memory:\s*([\d.]+)\s*(GB|TB)", hardware, re.MULTILINE)
        if mem_match and memory_bytes is None:
            value = float(mem_match.group(1))
            unit = mem_match.group(2)
            memory_bytes = int(value * (1024**4 if unit == "TB" else 1024**3))

    displays = _run(["system_profiler", "SPDisplaysDataType"])
    if displays:
        chipset_match = re.search(r"^\s*Chipset Model:\s*(.+)$", displays, re.MULTILINE)
        if chipset_match and not chip:
            chip = chipset_match.group(1).strip()
        match = re.search(r"Total Number of Cores:\s*(\d+)", displays)
        if match:
            gpu_cores = int(match.group(1))

    if not chip:
        chip = "Apple Silicon" if platform.system() == "Darwin" else platform.machine()

    apple_silicon_generation, apple_silicon_tier = parse_apple_silicon_chip(chip)
    return HardwareInfo(
        chip_name=chip,
        unified_memory_bytes=memory_bytes,
        gpu_cores=gpu_cores,
        apple_silicon_generation=apple_silicon_generation,
        apple_silicon_tier=apple_silicon_tier,
    )
