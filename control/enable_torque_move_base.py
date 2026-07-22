"""Standalone hardware test: enable torque and rotate the arm's base joint a few degrees.

Run directly, e.g.:
    python3 enable_torque_move_base.py
    python3 enable_torque_move_base.py --port /dev/ttyUSB0 --degrees -2
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from ..interface.dynamixel_driver import DynamixelDriver
from ..interface.ports import list_serial_ports, recommended_port

BASE_JOINT_INDEX = 0
DEFAULT_DEGREES = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="", help="Serial port (default: auto-detect)")
    parser.add_argument(
        "--degrees",
        type=float,
        default=DEFAULT_DEGREES,
        help=f"Degrees to rotate the base joint (default: {DEFAULT_DEGREES})",
    )
    args = parser.parse_args()

    port = args.port or recommended_port(list_serial_ports())
    if not port:
        raise SystemExit("No serial port found; pass --port explicitly (e.g. /dev/ttyUSB0)")

    driver = DynamixelDriver()
    print(f"Connecting to {port} ...")
    driver.connect(port)  # connect() initializes and enables torque on all motors
    print("Connected; torque enabled on all motors.")

    try:
        current_joints = driver.read_joint_angles()
        print(f"Current joint angles (rad): {current_joints}")

        target_joints = list(current_joints)
        target_joints[BASE_JOINT_INDEX] += math.radians(args.degrees)

        driver.set_joint_speeds([driver.default_speed] * 4)
        driver.set_joint_angles(target_joints)
        print(f"Base joint commanded to move by {args.degrees} degrees.")

        time.sleep(1.5)
        print(f"New joint angles (rad): {driver.read_joint_angles()}")
    except Exception:
        driver.disconnect()
        raise

    print("Done. Leaving the connection open so torque stays enabled and the arm holds position.")


if __name__ == "__main__":
    main()
