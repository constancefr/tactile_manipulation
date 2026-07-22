#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from dynamixel_driver import DynamixelDriver

"""
Experimentally find the gripper's fully-open and fully-closed tick values.

Two modes:

  stall (default) -- automated. Gently drives the gripper toward each
  mechanical extreme under a low current limit, so it stalls safely against
  the hard stop instead of slamming into it at full torque, then reads back
  wherever it actually settled.

      python3 gripper_calibrate.py --port /dev/ttyUSB0 --stall-current 30

  manual -- disables gripper torque only (arm joints keep holding their
  pose), and lets you push the gripper to each extreme by hand, recording the
  position when you press Enter.

      python3 gripper_calibrate.py --port /dev/ttyUSB0 --mode manual
"""

DXL_MIN_POSITION = 0
DXL_MAX_POSITION = 4095
POLL_INTERVAL_SEC = 0.2
STABLE_READS_REQUIRED = 4
STABLE_TOLERANCE_TICKS = 2
MAX_WAIT_SEC = 8.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate gripper open/closed tick values")
    p.add_argument("--port", required=True, help="Serial port for Dynamixel, e.g. /dev/ttyUSB0")
    p.add_argument("--mode", choices=["stall", "manual"], default="stall")
    p.add_argument(
        "--stall-current",
        type=int,
        default=30,
        help="Raw current-register units used to gently press against each mechanical limit (stall mode only). "
        "Start low and increase only if the gripper doesn't move at all.",
    )
    return p.parse_args()


def wait_for_settle(driver: DynamixelDriver) -> int:
    stable_count = 0
    last_position = driver.read_gripper_position()
    deadline = time.monotonic() + MAX_WAIT_SEC
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SEC)
        position = driver.read_gripper_position()
        if abs(position - last_position) <= STABLE_TOLERANCE_TICKS:
            stable_count += 1
            if stable_count >= STABLE_READS_REQUIRED:
                return position
        else:
            stable_count = 0
        last_position = position
    return last_position


def calibrate_stall(driver: DynamixelDriver, stall_current: int) -> None:
    print(f"Pressing toward closed (position={DXL_MIN_POSITION}) at max_current={stall_current}...")
    driver.set_gripper_position(DXL_MIN_POSITION, max_current=stall_current)
    closed_ticks = wait_for_settle(driver)
    closed_ma = abs(driver.read_gripper_current_ma())
    print(f"Closed settled at ticks={closed_ticks} current={closed_ma:.1f}mA")

    print(f"Pressing toward open (position={DXL_MAX_POSITION}) at max_current={stall_current}...")
    driver.set_gripper_position(DXL_MAX_POSITION, max_current=stall_current)
    open_ticks = wait_for_settle(driver)
    open_ma = abs(driver.read_gripper_current_ma())
    print(f"Open settled at ticks={open_ticks} current={open_ma:.1f}mA")

    print()
    print(f"Result: closed_ticks={closed_ticks}  open_ticks={open_ticks}")
    print("If closed/open current is pegged near your --stall-current the whole time, it's genuinely")
    print("stalled against the hard stop. If current stayed near 0, the gripper reached the commanded")
    print("position without hitting a limit -- widen the requested range and retry.")


def calibrate_manual(driver: DynamixelDriver) -> None:
    driver.set_gripper_torque(False)
    print("Gripper torque disabled (arm joints unaffected). Move the gripper by hand.")
    try:
        input("Move to fully CLOSED, then press Enter...")
        closed_ticks = driver.read_gripper_position()
        print(f"Closed recorded at ticks={closed_ticks}")

        input("Move to fully OPEN, then press Enter...")
        open_ticks = driver.read_gripper_position()
        print(f"Open recorded at ticks={open_ticks}")
    finally:
        driver.set_gripper_torque(True)
        print("Gripper torque re-enabled.")

    print()
    print(f"Result: closed_ticks={closed_ticks}  open_ticks={open_ticks}")


def main() -> None:
    args = parse_args()
    driver = DynamixelDriver()
    print(f"Connecting to {args.port}...")
    driver.connect(args.port)
    try:
        if args.mode == "stall":
            calibrate_stall(driver, args.stall_current)
        else:
            calibrate_manual(driver)
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
