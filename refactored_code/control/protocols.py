"""Structural interfaces used by the control layer.

The concrete Dynamixel driver lives in the project's ``interface`` package.
Keeping that dependency behind a Protocol makes the control code testable
without connected hardware.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class DynamixelDriverProtocol(Protocol):
    """Methods required by the arm and gripper controllers."""

    default_speed: int

    def connect(self, port: str) -> None: ...

    def disconnect(self) -> None: ...

    def close_port_keep_torque(self) -> None: ...

    def read_joint_angles(self) -> Sequence[float]: ...

    def set_joint_angles(self, angles: Sequence[float]) -> None: ...

    def set_joint_speeds(self, speeds: Sequence[int]) -> None: ...

    def read_gripper_position(self) -> int: ...

    def set_gripper_position(
        self, position: int, max_current: int | None = None
    ) -> None: ...

    def set_gripper_current(self, current: int) -> None: ...

    def read_gripper_current_ma(self) -> float: ...
