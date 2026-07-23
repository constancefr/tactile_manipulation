"""Reusable high-level gripper controller."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .protocols import DynamixelDriverProtocol


@dataclass(frozen=True)
class GripperCalibration:
    """Calibrated mechanical limits in raw Dynamixel ticks."""

    closed_ticks: int
    open_ticks: int
    minimum_ticks: int = 0
    maximum_ticks: int = 4095

    def __post_init__(self) -> None:
        if self.minimum_ticks >= self.maximum_ticks:
            raise ValueError("minimum_ticks must be smaller than maximum_ticks")
        for name, value in (
            ("closed_ticks", self.closed_ticks),
            ("open_ticks", self.open_ticks),
        ):
            if not self.minimum_ticks <= value <= self.maximum_ticks:
                raise ValueError(
                    f"{name}={value} is outside "
                    f"[{self.minimum_ticks}, {self.maximum_ticks}]"
                )

    @property
    def lower_mechanical_limit(self) -> int:
        return min(self.closed_ticks, self.open_ticks)

    @property
    def upper_mechanical_limit(self) -> int:
        return max(self.closed_ticks, self.open_ticks)

    @property
    def opening_direction(self) -> int:
        return 1 if self.open_ticks >= self.closed_ticks else -1


@dataclass(frozen=True)
class GripperState:
    position_ticks: int
    current_ma: float


class GripperController:
    """Position- and current-based gripper operations.

    The controller shares an already-connected driver with the arm controller.
    """

    def __init__(
        self,
        driver: DynamixelDriverProtocol,
        calibration: GripperCalibration,
        *,
        settle_delay_sec: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if settle_delay_sec < 0.0:
            raise ValueError("settle_delay_sec cannot be negative")
        self.driver = driver
        self.calibration = calibration
        self.settle_delay_sec = settle_delay_sec
        self._sleep = sleep

    def open(self, *, max_current: int | None = None) -> GripperState:
        return self.move_to(self.calibration.open_ticks, max_current=max_current)

    def close(self, *, max_current: int | None = None) -> GripperState:
        return self.move_to(self.calibration.closed_ticks, max_current=max_current)

    def move_to(
        self,
        position_ticks: int,
        *,
        max_current: int | None = None,
        settle: bool = True,
    ) -> GripperState:
        target = self._validate_position(position_ticks)
        if max_current is not None and max_current <= 0:
            raise ValueError("max_current must be positive when supplied")
        self.driver.set_gripper_position(target, max_current=max_current)
        if settle and self.settle_delay_sec:
            self._sleep(self.settle_delay_sec)
        return self.read_state()

    def apply_current(
        self,
        current: int,
        *,
        settle: bool = True,
    ) -> GripperState:
        if current == 0:
            raise ValueError("current must be non-zero")
        self.driver.set_gripper_current(int(current))
        if settle and self.settle_delay_sec:
            self._sleep(self.settle_delay_sec)
        return self.read_state()

    def jog(
        self,
        *,
        opening: bool,
        step_ticks: int,
        max_current: int | None = None,
    ) -> GripperState:
        if step_ticks <= 0:
            raise ValueError("step_ticks must be positive")
        current_position = int(self.driver.read_gripper_position())
        direction = self.calibration.opening_direction
        if not opening:
            direction *= -1
        target = current_position + direction * step_ticks
        target = max(
            self.calibration.lower_mechanical_limit,
            min(self.calibration.upper_mechanical_limit, target),
        )
        return self.move_to(target, max_current=max_current, settle=False)

    def read_state(self) -> GripperState:
        return GripperState(
            position_ticks=int(self.driver.read_gripper_position()),
            current_ma=abs(float(self.driver.read_gripper_current_ma())),
        )

    def _validate_position(self, position_ticks: int) -> int:
        value = int(position_ticks)
        if not (
            self.calibration.lower_mechanical_limit
            <= value
            <= self.calibration.upper_mechanical_limit
        ):
            raise ValueError(
                f"Gripper target {value} is outside calibrated mechanical limits "
                f"[{self.calibration.lower_mechanical_limit}, "
                f"{self.calibration.upper_mechanical_limit}]"
            )
        return value
