"""Reusable high-level arm motion controller."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .kinematics import (
    ArmKinematics,
    ArmPose,
    interpolate_joint_steps,
    synchronized_joint_speeds,
)
from .protocols import DynamixelDriverProtocol


@dataclass(frozen=True)
class Workspace:
    """Conservative Cartesian workspace, expressed in metres."""

    min_x: float = 0.0
    max_x: float = 0.30
    min_y: float = -0.30
    max_y: float = 0.30
    min_z: float = 0.0
    max_z: float = 0.30

    def contains(self, pose: ArmPose) -> bool:
        return (
            self.min_x <= pose.x <= self.max_x
            and self.min_y <= pose.y <= self.max_y
            and self.min_z <= pose.z <= self.max_z
        )

    def validate(self, pose: ArmPose) -> None:
        if not self.contains(pose):
            raise ValueError(
                "Requested pose lies outside the configured workspace: "
                f"{pose}; x=[{self.min_x}, {self.max_x}], "
                f"y=[{self.min_y}, {self.max_y}], "
                f"z=[{self.min_z}, {self.max_z}] metres"
            )


@dataclass(frozen=True)
class MotionConfig:
    max_speed: int = 100
    min_speed: int = 5
    max_step_deg: float = 100.0
    step_delay_sec: float = 0.001
    ik_position_tolerance_m: float = 1e-6
    ik_angle_tolerance_rad: float = 1e-6

    def __post_init__(self) -> None:
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if not 0 < self.min_speed <= self.max_speed:
            raise ValueError("min_speed must be in [1, max_speed]")
        if self.max_step_deg <= 0.0:
            raise ValueError("max_step_deg must be positive")
        if self.step_delay_sec < 0.0:
            raise ValueError("step_delay_sec cannot be negative")


@dataclass(frozen=True)
class ArmMoveResult:
    target_pose: ArmPose
    target_joints: tuple[float, float, float, float]
    final_joints: tuple[float, float, float, float]
    waypoint_count: int


class ArmController:
    """High-level Cartesian and joint-space operations for the robot arm.

    The controller does not own the serial connection. A connected driver is
    injected so the same connection can be shared with ``GripperController``.
    """

    BASE_JOINT_INDEX = 0

    def __init__(
        self,
        driver: DynamixelDriverProtocol,
        *,
        kinematics: ArmKinematics | None = None,
        workspace: Workspace | None = None,
        motion: MotionConfig | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.driver = driver
        self.kinematics = kinematics or ArmKinematics()
        self.workspace = workspace or Workspace()
        self.motion = motion or MotionConfig()
        self._sleep = sleep

    def solve_pose(self, pose: ArmPose) -> tuple[float, float, float, float]:
        """Validate a pose and return its inverse-kinematics solution."""
        self.workspace.validate(pose)
        target_joints = self.kinematics.inverse(pose)
        self._validate_round_trip(pose, target_joints)
        return target_joints

    def move_to_pose(self, pose: ArmPose) -> ArmMoveResult:
        """Move to an absolute Cartesian pose using smoothed joint waypoints."""
        target_joints = self.solve_pose(pose)
        current_joints = self.read_joint_angles()
        waypoints = interpolate_joint_steps(
            current_joints,
            target_joints,
            self.motion.max_step_deg,
        )

        for waypoint in waypoints:
            speeds = synchronized_joint_speeds(
                current_joints,
                waypoint,
                maximum_speed=self.motion.max_speed,
                minimum_speed=self.motion.min_speed,
            )
            self.driver.set_joint_speeds(speeds)
            self.driver.set_joint_angles(waypoint)
            if self.motion.step_delay_sec:
                self._sleep(self.motion.step_delay_sec)
            current_joints = self.read_joint_angles()

        return ArmMoveResult(
            target_pose=pose,
            target_joints=target_joints,
            final_joints=current_joints,
            waypoint_count=len(waypoints),
        )

    def move_to_cm(
        self,
        x_cm: float,
        y_cm: float,
        z_cm: float,
        pitch_deg: float = 0.0,
    ) -> ArmMoveResult:
        """Convenience wrapper around :meth:`move_to_pose`."""
        return self.move_to_pose(
            ArmPose.from_cm_degrees(x_cm, y_cm, z_cm, pitch_deg)
        )

    def rotate_base_by(
        self,
        degrees: float,
        *,
        settle_delay_sec: float = 1.5,
    ) -> tuple[float, float, float, float]:
        """Rotate only the base joint relative to its current angle."""
        if not math.isfinite(degrees):
            raise ValueError("degrees must be finite")
        if settle_delay_sec < 0.0:
            raise ValueError("settle_delay_sec cannot be negative")

        target = list(self.read_joint_angles())
        target[self.BASE_JOINT_INDEX] += math.radians(degrees)
        self.driver.set_joint_speeds([self.motion.max_speed] * 4)
        self.driver.set_joint_angles(target)
        if settle_delay_sec:
            self._sleep(settle_delay_sec)
        return self.read_joint_angles()

    def read_joint_angles(self) -> tuple[float, float, float, float]:
        values = tuple(float(value) for value in self.driver.read_joint_angles()[:4])
        if len(values) != 4:
            raise RuntimeError("Driver returned fewer than four joint angles")
        return values

    def read_pose(self) -> ArmPose:
        return self.kinematics.forward(self.read_joint_angles())

    def _validate_round_trip(
        self,
        target_pose: ArmPose,
        target_joints: Sequence[float],
    ) -> None:
        solved_pose = self.kinematics.forward(target_joints)
        position_error = math.dist(
            (solved_pose.x, solved_pose.y, solved_pose.z),
            (target_pose.x, target_pose.y, target_pose.z),
        )
        angle_error = abs(
            math.atan2(
                math.sin(solved_pose.angle_rad - target_pose.angle_rad),
                math.cos(solved_pose.angle_rad - target_pose.angle_rad),
            )
        )
        if (
            position_error > self.motion.ik_position_tolerance_m
            or angle_error > self.motion.ik_angle_tolerance_rad
        ):
            raise RuntimeError(
                "Forward/inverse kinematics round-trip mismatch: "
                f"position={position_error:.9f} m, "
                f"angle={angle_error:.9f} rad"
            )
