"""Standalone hardware test: verify forward kinematics by hand-moving the arm.

Disables torque so the arm can be moved freely by hand, then continuously reads
joint angles and prints the resulting end-effector (TCP) pose from
ArmKinematics.forward(), so you can sanity-check the kinematics against where
you actually see the arm.

Run directly, e.g.:
    python3 forward_kinematics_monitor.py --port /dev/ttyUSB0
    python3 forward_kinematics_monitor.py --rate 5

Press Ctrl+C to stop.
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

from .dynamixel_driver import DynamixelDriver
from .kinematics import ArmKinematics
from .ports import list_serial_ports, recommended_port

DEFAULT_RATE_HZ = 5.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="", help="Serial port (default: auto-detect)")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ, help=f"Poll rate in Hz (default: {DEFAULT_RATE_HZ})")
    args = parser.parse_args()

    port = args.port or recommended_port(list_serial_ports())
    if not port:
        raise SystemExit("No serial port found; pass --port explicitly (e.g. /dev/ttyUSB0)")

    period = 1.0 / max(0.1, args.rate)
    kinematics = ArmKinematics()
    driver = DynamixelDriver()
    print(f"Connecting to {port} ...")
    driver.connect(port)
    driver.disable_torque()
    print("Connected; torque disabled -- you can now move the arm by hand.")
    print("Reading joint angles and printing the TCP pose. Press Ctrl+C to stop.\n")

    try:
        while True:
            joints = driver.read_joint_angles()
            pose = kinematics.forward(joints)
            joints_deg = [math.degrees(j) for j in joints]
            print(
                f"\rTCP: x={pose.x:+.4f} y={pose.y:+.4f} z={pose.z:+.4f} "
                f"angle={math.degrees(pose.angle_rad):+7.2f} deg   "
                f"| joints(deg): [{', '.join(f'{j:+7.2f}' for j in joints_deg)}]   ",
                end="",
                flush=True,
            )
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        driver.disconnect()
        print("Disconnected (torque left disabled).")


if __name__ == "__main__":
    main()
