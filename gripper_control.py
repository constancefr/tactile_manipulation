#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty

from dynamixel_driver import DynamixelDriver

"""
Control the gripper by position, force-limited position, pure current (force),
or live keyboard jogging.

Examples:
    # Plain position control (full torque)
    python3 gripper_control.py --port /dev/ttyUSB0 --position 1800

    # Position control capped at a max current, so it stalls safely on contact
    # instead of crushing whatever it grips
    python3 gripper_control.py --port /dev/ttyUSB0 --position 1800 --max-current 60

    # Pure current/force control -- squeeze at a given force, no position target
    python3 gripper_control.py --port /dev/ttyUSB0 --current 80

    # Jog live with 'a'/'d' (or Left/Right arrow keys), 'q' or Esc to quit
    python3 gripper_control.py --port /dev/ttyUSB0 --interactive --step 20 --max-current 60

    # Fully close/open using the calibrated limits below
    python3 gripper_control.py --port /dev/ttyUSB0 --close
    python3 gripper_control.py --port /dev/ttyUSB0 --open
"""

DXL_MIN_POSITION = 0
DXL_MAX_POSITION = 4095

# Actual mechanical limits, in raw ticks -- do NOT assume these are 0/4095.
# Find the real values for your hardware with:
#   python3 gripper_calibrate.py --port /dev/ttyUSB0 --stall-current 30
# and copy its reported closed_ticks/open_ticks here. Until then, these
# placeholders just fall back to the (probably wrong) full range.
GRIPPER_CLOSED_TICKS = DXL_MIN_POSITION
GRIPPER_OPEN_TICKS = DXL_MAX_POSITION

SETTLE_DELAY_SEC = 1.0
DEFAULT_STEP = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control gripper by position, current, or live keyboard jog")
    p.add_argument("--port", required=True, help="Serial port for Dynamixel, e.g. /dev/ttyUSB0")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--position", type=int, help=f"Target gripper position [{DXL_MIN_POSITION},{DXL_MAX_POSITION}]")
    mode.add_argument("--current", type=int, help="Target gripper current (raw units, pure current control)")
    mode.add_argument("--interactive", action="store_true", help="Jog the gripper live with Left/Right arrow keys")
    mode.add_argument(
        "--close", action="store_true",
        help=f"Move to the calibrated fully-closed position (GRIPPER_CLOSED_TICKS={GRIPPER_CLOSED_TICKS})",
    )
    mode.add_argument(
        "--open", action="store_true",
        help=f"Move to the calibrated fully-open position (GRIPPER_OPEN_TICKS={GRIPPER_OPEN_TICKS})",
    )
    p.add_argument(
        "--max-current",
        type=int,
        default=None,
        help="With --position/--interactive: cap current so the gripper stalls on contact "
        "instead of crushing the object",
    )
    p.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help="Ticks moved per arrow-key press in --interactive mode",
    )
    return p.parse_args()


def read_key(fd: int) -> str | None:
    '''
    Blocks for a single keypress. Returns "LEFT", "RIGHT", "QUIT", or None for
    any other key. Assumes the terminal is already in cbreak mode.

    Reads via os.read() on the raw fd rather than sys.stdin.read(), because
    sys.stdin is buffered: a single sys.stdin.read(1) can silently pull all
    bytes of an arrow-key escape sequence out of the OS pipe at once, after
    which select.select() on sys.stdin sees nothing left pending (it only
    checks the OS-level fd, not Python's internal buffer) and the escape
    sequence gets misread as a bare Escape press.
    '''
    ch = os.read(fd, 1).decode(errors="ignore")
    if ch in ("a", "A"):
        return "LEFT"
    if ch in ("d", "D"):
        return "RIGHT"
    if ch in ("q", "Q"):
        return "QUIT"
    if ch != "\x1b":
        return None
    # Esc alone (no further bytes arrive quickly) means quit; Esc followed by
    # "[X" is an arrow-key escape sequence.
    if not select.select([fd], [], [], 0.1)[0]:
        return "QUIT"
    ch2 = os.read(fd, 1).decode(errors="ignore")
    if ch2 != "[" or not select.select([fd], [], [], 0.1)[0]:
        return None
    ch3 = os.read(fd, 1).decode(errors="ignore")
    if ch3 == "C":
        return "RIGHT"
    if ch3 == "D":
        return "LEFT"
    return None


def run_interactive(driver: DynamixelDriver, step: int, max_current: int | None) -> None:
    position = driver.read_gripper_position()
    print(f"Interactive gripper jog. 'a'/Left=close, 'd'/Right=open, step={step} ticks, 'q'/Esc to quit.")
    print(f"Starting position: {position}")

    # Direction of "open" in tick-space -- handles GRIPPER_CLOSED_TICKS being
    # either smaller or larger than GRIPPER_OPEN_TICKS, since that depends on
    # this motor's mounting/rotation direction.
    open_dir = 1 if GRIPPER_OPEN_TICKS >= GRIPPER_CLOSED_TICKS else -1
    lo, hi = sorted((GRIPPER_CLOSED_TICKS, GRIPPER_OPEN_TICKS))

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = read_key(fd)
            if key == "QUIT":
                break
            if key not in ("LEFT", "RIGHT"):
                continue
            delta = (step if key == "RIGHT" else -step) * open_dir
            position = max(lo, min(hi, position + delta))
            driver.set_gripper_position(position, max_current=max_current)
            time.sleep(0.05)
            actual = driver.read_gripper_position()
            # present_current is signed (direction the motor is pushing, not just
            # magnitude) -- abs() here since only the force magnitude matters for jogging.
            current_ma = abs(driver.read_gripper_current_ma())
            print(f"\rtarget={position:4d}  actual={actual:4d}  current={current_ma:6.1f}mA   ", end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()


def main() -> None:
    args = parse_args()
    driver = DynamixelDriver()
    print(f"Connecting to {args.port}...")
    driver.connect(args.port)
    try:
        if args.interactive:
            run_interactive(driver, args.step, args.max_current)
            return

        if args.close or args.open:
            target = GRIPPER_CLOSED_TICKS if args.close else GRIPPER_OPEN_TICKS
            label = "closed" if args.close else "open"
            print(f"Commanding gripper to calibrated fully-{label} position={target} (max_current={args.max_current})")
            driver.set_gripper_position(target, max_current=args.max_current)
        elif args.position is not None:
            mode = "current-based position" if args.max_current is not None else "position"
            print(f"Commanding gripper to position={args.position} ({mode} control, max_current={args.max_current})")
            driver.set_gripper_position(args.position, max_current=args.max_current)
        else:
            print(f"Commanding gripper current={args.current} (pure current control)")
            driver.set_gripper_current(args.current)

        time.sleep(SETTLE_DELAY_SEC)
        position = driver.read_gripper_position()
        current_ma = abs(driver.read_gripper_current_ma())
        print(f"Settled: position={position} current={current_ma:.1f}mA")
    finally:
        try:
            driver.close_port_keep_torque()
        except Exception:
            try:
                driver.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
