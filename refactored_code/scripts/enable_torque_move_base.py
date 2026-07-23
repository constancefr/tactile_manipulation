#!/usr/bin/env python3
"""Connect, enable torque, and rotate the arm base by a small angle."""

from __future__ import annotations

import argparse
import math

from control import ArmController
from interface.dynamixel_driver import DynamixelDriver
from interface.ports import list_serial_ports, recommended_port


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="", help="Default: auto-detect")
    parser.add_argument("--degrees", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    port = args.port or recommended_port(list_serial_ports())
    if not port:
        raise SystemExit("No serial port found; pass --port explicitly")

    driver = DynamixelDriver()
    driver.connect(port)
    try:
        joints = ArmController(driver).rotate_base_by(args.degrees)
        print("Joint angles (deg):", [round(math.degrees(v), 3) for v in joints])
    finally:
        driver.close_port_keep_torque()


if __name__ == "__main__":
    main()
