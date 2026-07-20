"""Standalone hardware test: check inverse kinematics against a hand-moved pose.

Disables torque so the arm can be moved freely by hand, and continuously prints
the live TCP pose from ArmKinematics.forward() -- same idea as
forward_kinematics_monitor.py. Press Enter at any time to snapshot the current
joints/pose and run ArmKinematics.inverse() on that pose, then report two
things:

  1. Round-trip pose error: forward(inverse(pose)) vs the original pose.
     This should be ~0 always -- it's the real correctness check on inverse().
  2. Joint-space difference: inverse(pose) vs the joints actually measured.
     inverse() always returns one specific elbow configuration, so if the arm
     is hand-posed in the mirrored elbow solution this will show a large
     difference even though the pose itself matches -- that's expected, not a
     bug, and this check makes that visible.

Run directly, e.g.:
    python3 ik_round_trip_check.py --port /dev/cu.usbserial-FT88Z15T 

Press Enter to check, Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .dynamixel_driver import DynamixelDriver
from .kinematics import ArmKinematics, ArmPose
from .ports import list_serial_ports, recommended_port

DEFAULT_RATE_HZ = 5.0


def _watch_for_enter(stop_event: threading.Event, trigger_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            input()
        except EOFError:
            return
        trigger_event.set()


def _format_pose(pose: ArmPose) -> str:
    return f"x={pose.x:+.4f} y={pose.y:+.4f} z={pose.z:+.4f} angle={math.degrees(pose.angle_rad):+7.2f} deg"


def _run_ik_check(kinematics: ArmKinematics, joints: list[float], pose: ArmPose) -> None:
    print("\n--- inverse kinematics check ---")
    print(f"Measured pose:   {_format_pose(pose)}")
    print(f"Measured joints (deg): [{', '.join(f'{math.degrees(j):+7.2f}' for j in joints)}]")
    try:
        ik_joints = kinematics.inverse(pose)
    except ValueError as exc:
        print(f"inverse() raised: {exc}")
        print("--- resuming monitor ---\n")
        return

    check_pose = kinematics.forward(ik_joints)
    pose_error = (
        check_pose.x - pose.x,
        check_pose.y - pose.y,
        check_pose.z - pose.z,
        math.degrees(check_pose.angle_rad - pose.angle_rad),
    )
    joint_diff_deg = [math.degrees(a - b) for a, b in zip(ik_joints, joints)]

    print(f"IK joints (deg):       [{', '.join(f'{math.degrees(j):+7.2f}' for j in ik_joints)}]")
    print(
        "Round-trip pose error (forward(inverse(pose)) - pose): "
        f"dx={pose_error[0]:+.6f} dy={pose_error[1]:+.6f} dz={pose_error[2]:+.6f} dangle={pose_error[3]:+.4f} deg"
    )
    print(f"Joint diff vs measured (deg): [{', '.join(f'{d:+7.2f}' for d in joint_diff_deg)}]")
    if any(abs(d) > 1.0 for d in joint_diff_deg):
        print("Note: large joint diff with small pose error usually means the arm is in the")
        print("mirrored elbow configuration -- inverse() only returns one branch.")
    print("--- resuming monitor ---\n")


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
    print("Press Enter to run an inverse-kinematics check on the current pose. Ctrl+C to stop.\n")

    stop_event = threading.Event()
    trigger_event = threading.Event()
    watcher = threading.Thread(target=_watch_for_enter, args=(stop_event, trigger_event), daemon=True)
    watcher.start()

    try:
        while True:
            joints = driver.read_joint_angles()
            pose = kinematics.forward(joints)
            print(f"\rTCP: {_format_pose(pose)}   ", end="", flush=True)
            if trigger_event.is_set():
                trigger_event.clear()
                _run_ik_check(kinematics, joints, pose)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_event.set()
        driver.disconnect()
        print("Disconnected (torque left disabled).")


if __name__ == "__main__":
    main()
