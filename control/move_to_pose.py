#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Sequence

from control.kinematics import ArmKinematics, ArmPose, synchronized_joint_speeds
from interface.dynamixel_driver import DynamixelDriver

"""
Move end-effector to absolute Cartesian position (cm, degrees)
To run:
    python move_to_pose.py --x-cm 20 --y-cm 12.5 --z-cm 10 --pitch-deg 0 --port /dev/cu.usbserial-FT88Z15T
"""

# Allowed workspace in centimetres
# TODO: check workspace limits!!
MIN_X_CM = 0.0
MAX_X_CM = 30.0
MIN_Y_CM = -30.0
MAX_Y_CM = 30.0
MIN_Z_CM = 0.0
MAX_Z_CM = 30.0

# Motion smoothing
MAX_STEP_DEG = 2.0  # maximum joint angle change per interpolation step (degrees)
STEP_DELAY_SEC = 0.2  # delay between intermediate commands
# STEP_DELAY_SEC = 0.15  # delay between intermediate commands
DEFAULT_MAX_SPEED = 60
MIN_SPEED = 5

IK_POS_TOL = 1e-6
IK_ANG_TOL = 1e-6


def finite_float(value: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError("value must be finite")
    return v


def in_workspace(x_cm: float, y_cm: float, z_cm: float) -> bool:
    return (
        MIN_X_CM <= x_cm <= MAX_X_CM
        and MIN_Y_CM <= y_cm <= MAX_Y_CM
        and MIN_Z_CM <= z_cm <= MAX_Z_CM
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Move end-effector to absolute Cartesian position (cm, degrees)")
    p.add_argument("--x-cm", type=finite_float, required=True, help=f"X in cm [{MIN_X_CM},{MAX_X_CM}]")
    p.add_argument("--y-cm", type=finite_float, required=True, help=f"Y in cm [{MIN_Y_CM},{MAX_Y_CM}]")
    p.add_argument("--z-cm", type=finite_float, required=True, help=f"Z in cm [{MIN_Z_CM},{MAX_Z_CM}]")
    p.add_argument("--pitch-deg", type=finite_float, default=0.0, help="End-effector pitch in degrees (optional)")
    p.add_argument("--port", default="", help="Serial port for Dynamixel (omit for dry-run)")
    p.add_argument("--max-speed", type=int, default=DEFAULT_MAX_SPEED, help="Maximum joint speed for synchronized speeds")
    p.add_argument("--step-deg", type=float, default=MAX_STEP_DEG, help="Max joint change per interpolation step in degrees")
    return p.parse_args()


def cm_deg_to_pose(x_cm: float, y_cm: float, z_cm: float, pitch_deg: float) -> ArmPose:
    return ArmPose(x_cm * 0.01, y_cm * 0.01, z_cm * 0.01, math.radians(pitch_deg))


def validate_round_trip(kin: ArmKinematics, target_pose: ArmPose, target_joints: Sequence[float]) -> None:
    '''
    Validates that the computed joint angles produce the expected pose when passed through forward kinematics.
    '''
    solved = kin.forward(target_joints)
    pos_err = math.sqrt((solved.x - target_pose.x) ** 2 + (solved.y - target_pose.y) ** 2 + (solved.z - target_pose.z) ** 2)
    ang_err = abs(math.atan2(math.sin(solved.angle_rad - target_pose.angle_rad), math.cos(solved.angle_rad - target_pose.angle_rad)))
    if pos_err > IK_POS_TOL or ang_err > IK_ANG_TOL:
        raise RuntimeError(f"FK round-trip mismatch: pos {pos_err:.9f} m ang {ang_err:.9f} rad")


def interpolate_joint_steps(start: Sequence[float], end: Sequence[float], max_step_deg: float) -> list[list[float]]:
    '''
    Interpolates joint angles from start to end, returning a list of intermediate waypoints.
    Each waypoint is a list of joint angles in radians. The maximum change per joint per step is limited to max_step_deg.
    '''
    changes_deg = [abs(math.degrees(e - s)) for s, e in zip(start[:4], end[:4])]
    max_change = max(changes_deg)
    if max_change == 0.0:
        return []
    steps = max(1, math.ceil(max_change / max_step_deg))
    waypoints: list[list[float]] = []
    for i in range(1, steps + 1):
        frac = i / steps
        waypoint = [float(s + frac * (e - s)) for s, e in zip(start[:4], end[:4])]
        waypoints.append(waypoint)
    return waypoints


def main() -> None:
    args = parse_args()
    if not in_workspace(args.x_cm, args.y_cm, args.z_cm):
        raise SystemExit(f"Requested point outside allowed workspace in cm: x[{MIN_X_CM},{MAX_X_CM}], y[{MIN_Y_CM},{MAX_Y_CM}], z[{MIN_Z_CM},{MAX_Z_CM}]")

    kinematics = ArmKinematics()
    target_pose = cm_deg_to_pose(args.x_cm, args.y_cm, args.z_cm, args.pitch_deg)
    print(f"Target pose: x={target_pose.x:.4f} y={target_pose.y:.4f} z={target_pose.z:.4f} pitch_deg={args.pitch_deg:.2f}")

    try:
        target_joints = kinematics.inverse(target_pose)
    except Exception as exc:
        raise SystemExit(f"IK failed: {exc}") from exc

    validate_round_trip(kinematics, target_pose, target_joints)

    if not args.port:
        print("Dry-run: computed joint targets (radians):")
        print([f"{j:.6f}" for j in target_joints])
        return

    driver = DynamixelDriver(default_speed=args.max_speed)
    print(f"Connecting to {args.port}...")
    driver.connect(args.port)
    try:
        current_joints = driver.read_joint_angles()[:4]
        print("Current joints (deg):", [round(math.degrees(j),3) for j in current_joints])
        print("Target joints (deg):", [round(math.degrees(j),3) for j in target_joints])

        waypoints = interpolate_joint_steps(current_joints, target_joints, args.step_deg)
        if not waypoints:
            print("Already at target.")
            return

        for idx, waypoint in enumerate(waypoints, start=1):
            speeds = synchronized_joint_speeds(current_joints, waypoint, maximum_speed=args.max_speed, minimum_speed=MIN_SPEED)
            driver.set_joint_speeds(speeds)
            driver.set_joint_angles(waypoint)
            print(f"Step {idx}/{len(waypoints)} sent; waiting {STEP_DELAY_SEC}s")
            time.sleep(STEP_DELAY_SEC)
            current_joints = driver.read_joint_angles()[:4]

        print("Final readback joints (deg):", [round(math.degrees(j),3) for j in current_joints])
    finally:
        # Close the serial port but keep motor torque enabled so the arm holds
        # the final commanded position. If you prefer to disable torque here
        # instead, call `driver.disconnect()`.
        try:
            driver.close_port_keep_torque()
        except Exception:
            # Fall back to a full disconnect if closing the port fails
            try:
                driver.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
