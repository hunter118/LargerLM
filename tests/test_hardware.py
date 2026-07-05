from __future__ import annotations

from largerlm.hardware import parse_apple_silicon_chip


def test_parse_apple_silicon_chip_recognizes_m5_max() -> None:
    generation, tier = parse_apple_silicon_chip("Apple M5 Max")

    assert generation == 5
    assert tier == "Max"


def test_parse_apple_silicon_chip_handles_base_and_non_apple_names() -> None:
    assert parse_apple_silicon_chip("Apple M4") == (4, None)
    assert parse_apple_silicon_chip("Intel(R) Core(TM) i9") == (None, None)
