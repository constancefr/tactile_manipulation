"""Standalone hardware test: verify inverse kinematics from measured arm poses.

Disables torque so the arm can be moved freely by hand, then continuously reads
joint angles, computes the TCP pose with ArmKinematics.forward(), solves that
pose back to joint angles with ArmKinematics.inverse(), and prints both the
joint and pose round-trip errors.

Run directly, e.g.:
    python3 inverse_kinematics_monitor.py --port /dev/cu.usbserial-FT88Z15T
    python3 inverse_kinematics_monitor.py --rate 5

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

from ..interface.dynamixel_driver import DynamixelDriver
from .kinematics import ArmKinematics
from ..interface.ports import list_serial_ports, recommended_port

DEFAULT_RATE_HZ = 5.0


def angle_error_rad(actual: float, expected: float) -> float:
    return math.atan2(math.sin(actual - expected), math.cos(actual - expected))


def pose_error(actual, expected) -> tuple[float, float, float, float]:
    return (
        actual.x - expected.x,
        actual.y - expected.y,
        actual.z - expected.z,
        angle_error_rad(actual.angle_rad, expected.angle_rad),
    )


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
    print("Reading joints, solving FK -> IK -> FK. Press Ctrl+C to stop.\n")

    try:
        while True:
            joints = driver.read_joint_angles()[:4]
            pose = kinematics.forward(joints)
            try:
                ik_joints = kinematics.inverse(pose)
                ik_pose = kinematics.forward(ik_joints)
                joint_errors = [angle_error_rad(ik, measured) for ik, measured in zip(ik_joints, joints)]
                dx, dy, dz, dangle = pose_error(ik_pose, pose)
                print(
                    "\r"
                    f"TCP: x={pose.x:+.4f} y={pose.y:+.4f} z={pose.z:+.4f} "
                    f"angle={math.degrees(pose.angle_rad):+7.2f} deg   "
                    f"| IK joints(deg): [{', '.join(f'{math.degrees(j):+7.2f}' for j in ik_joints)}]   "
                    f"| joint err(deg): [{', '.join(f'{math.degrees(e):+7.3f}' for e in joint_errors)}]   "
                    f"| pose err: dx={dx:+.5f} dy={dy:+.5f} dz={dz:+.5f} "
                    f"da={math.degrees(dangle):+7.3f} deg   ",
                    end="",
                    flush=True,
                )
            except ValueError as exc:
                print(
                    "\r"
                    f"TCP: x={pose.x:+.4f} y={pose.y:+.4f} z={pose.z:+.4f} "
                    f"angle={math.degrees(pose.angle_rad):+7.2f} deg   "
                    f"| inverse failed: {exc}   ",
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
