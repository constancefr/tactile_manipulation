#!/usr/bin/env python3
"""Control or interactively jog the gripper."""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import tty

from control import GripperCalibration, GripperController
from interface.dynamixel_driver import DynamixelDriver

# Replace these after calibration. Keeping them explicit prevents accidental
# assumptions that the mechanical limits are always 0 and 4095.
GRIPPER_CLOSED_TICKS = 0
GRIPPER_OPEN_TICKS = 4095


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--position", type=int)
    modes.add_argument("--current", type=int)
    modes.add_argument("--interactive", action="store_true")
    modes.add_argument("--close", action="store_true")
    modes.add_argument("--open", action="store_true")
    parser.add_argument("--max-current", type=int)
    parser.add_argument("--step", type=int, default=20)
    return parser.parse_args()


def read_key(fd: int) -> str | None:
    character = os.read(fd, 1).decode(errors="ignore")
    if character in ("a", "A"):
        return "CLOSE"
    if character in ("d", "D"):
        return "OPEN"
    if character in ("q", "Q"):
        return "QUIT"
    if character != "\x1b":
        return None
    if not select.select([fd], [], [], 0.1)[0]:
        return "QUIT"
    second = os.read(fd, 1).decode(errors="ignore")
    if second != "[" or not select.select([fd], [], [], 0.1)[0]:
        return None
    third = os.read(fd, 1).decode(errors="ignore")
    return {"C": "OPEN", "D": "CLOSE"}.get(third)


def run_interactive(
    controller: GripperController,
    *,
    step: int,
    max_current: int | None,
) -> None:
    fd = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(fd)
    print("'a'/Left=close, 'd'/Right=open, 'q'/Esc=quit")
    try:
        tty.setcbreak(fd)
        while True:
            key = read_key(fd)
            if key == "QUIT":
                return
            if key not in ("OPEN", "CLOSE"):
                continue
            state = controller.jog(
                opening=key == "OPEN",
                step_ticks=step,
                max_current=max_current,
            )
            print(
                f"\rposition={state.position_ticks:4d} "
                f"current={state.current_ma:6.1f}mA   ",
                end="",
                flush=True,
            )
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous_settings)
        print()


def main() -> None:
    args = parse_args()
    calibration = GripperCalibration(
        closed_ticks=GRIPPER_CLOSED_TICKS,
        open_ticks=GRIPPER_OPEN_TICKS,
    )
    driver = DynamixelDriver()
    driver.connect(args.port)
    controller = GripperController(driver, calibration)
    try:
        if args.interactive:
            run_interactive(
                controller,
                step=args.step,
                max_current=args.max_current,
            )
        elif args.close:
            print(controller.close(max_current=args.max_current))
        elif args.open:
            print(controller.open(max_current=args.max_current))
        elif args.position is not None:
            print(
                controller.move_to(
                    args.position,
                    max_current=args.max_current,
                )
            )
        else:
            print(controller.apply_current(args.current))
    finally:
        driver.close_port_keep_torque()


if __name__ == "__main__":
    main()
