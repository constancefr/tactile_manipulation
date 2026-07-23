from __future__ import annotations

from glob import glob
from pathlib import Path


def list_serial_ports() -> list[str]:
    candidates: list[str] = []
    for pattern in ("/dev/serial/by-id/*", "/dev/ttyUSB*", "/dev/ttyACM*"):
        candidates.extend(glob(pattern))
    unique = sorted({str(Path(path).resolve()) if path.startswith("/dev/serial/by-id") else path for path in candidates})
    return sorted(unique, key=_port_sort_key)


def recommended_port(ports: list[str]) -> str:
    return ports[0] if ports else ""


def _port_sort_key(path: str) -> tuple[int, str]:
    if "ttyUSB" in path:
        return (0, path)
    if "ttyACM" in path:
        return (1, path)
    return (2, path)
