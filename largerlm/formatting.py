from __future__ import annotations


def format_bytes(num_bytes: float | int) -> str:
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0


def format_rate_gbps(bytes_per_second: float | int) -> str:
    return f"{float(bytes_per_second) / (1024**3):.2f} GiB/s"
