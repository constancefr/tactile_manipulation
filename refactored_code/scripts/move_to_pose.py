#!/usr/bin/env python3
"""Move the arm end effector to an absolute Cartesian pose."""

from __future__ import annotations

import argparse
import math

from control import ArmController, ArmPose, MotionConfig
from interface.dynamixel_driver import DynamixelDriver


def finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-cm", type=finite_float, required=True)
    parser.add_argument("--y-cm", type=finite_float, required=True)
    parser.add_argument("--z-cm", type=finite_float, required=True)
    parser.add_argument("--pitch-deg", type=finite_float, default=0.0)
    parser.add_argument("--port", default="", help="Omit for an IK dry run")
    parser.add_argument("--max-speed", type=int, default=60)
    parser.add_argument("--step-deg", type=finite_float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose = ArmPose.from_cm_degrees(
        args.x_cm, args.y_cm, args.z_cm, args.pitch_deg
    )
    motion = MotionConfig(max_speed=args.max_speed, max_step_deg=args.step_deg)

    if not args.port:
        controller = ArmController(DynamixelDriver(), motion=motion)
        joints = controller.solve_pose(pose)
        print("Dry-run joint targets (rad):", [f"{value:.6f}" for value in joints])
        return

    driver = DynamixelDriver(default_speed=args.max_speed)
    driver.connect(args.port)
    try:
        result = ArmController(driver, motion=motion).move_to_pose(pose)
        print("Target joints (deg):", [round(math.degrees(v), 3) for v in result.target_joints])
        print("Final joints (deg):", [round(math.degrees(v), 3) for v in result.final_joints])
        print(f"Waypoints: {result.waypoint_count}")
    finally:
        driver.close_port_keep_torque()


if __name__ == "__main__":
    main()
