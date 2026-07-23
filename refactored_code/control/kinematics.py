"""Forward/inverse kinematics and joint-motion utility functions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ArmPose:
    """Cartesian end-effector pose, expressed in metres and radians."""

    x: float
    y: float
    z: float
    angle_rad: float

    @classmethod
    def from_cm_degrees(
        cls, x_cm: float, y_cm: float, z_cm: float, pitch_deg: float = 0.0
    ) -> "ArmPose":
        return cls(
            x=x_cm * 0.01,
            y=y_cm * 0.01,
            z=z_cm * 0.01,
            angle_rad=math.radians(pitch_deg),
        )


@dataclass(frozen=True)
class ArmGeometry:
    """Geometric and joint-limit parameters for the four-joint arm."""

    l1: float = 0.080
    a2: float = 0.130
    a3: float = 0.124
    a4: float = 0.111
    joint4_min_rad: float = math.radians(-110.0)
    joint4_max_rad: float = math.radians(90.0)

    @property
    def phi(self) -> float:
        return -math.atan2(0.024, 0.128)

    @property
    def angle_offsets(self) -> tuple[float, float, float, float]:
        offset = math.pi / 2 + self.phi
        return (0.0, offset, -offset, 0.0)


class ArmKinematics:
    """Forward and inverse kinematics for the arm."""

    def __init__(self, geometry: ArmGeometry | None = None) -> None:
        self.geometry = geometry or ArmGeometry()

    def inverse(self, pose: ArmPose) -> tuple[float, float, float, float]:
        """Convert an end-effector pose to four joint angles in radians."""
        g = self.geometry
        q1 = math.atan2(pose.y, pose.x)
        radius = math.hypot(pose.x, pose.y)
        z_relative = pose.z - g.l1
        wrist_radius = radius - g.a4 * math.cos(pose.angle_rad)
        wrist_z = z_relative - g.a4 * math.sin(pose.angle_rad)

        distance_squared = wrist_radius**2 + wrist_z**2
        cos_q3 = (
            distance_squared - g.a2**2 - g.a3**2
        ) / (2 * g.a2 * g.a3)
        if not -1.0 <= cos_q3 <= 1.0:
            raise ValueError(f"Pose is outside the reachable workspace: {pose}")

        # Preserve the elbow configuration used by the original implementation.
        q3 = -math.acos(max(-1.0, min(1.0, cos_q3)))
        alpha = math.atan2(wrist_z, wrist_radius)
        beta = math.atan2(
            g.a3 * math.sin(q3),
            g.a2 + g.a3 * math.cos(q3),
        )
        q2 = alpha - beta
        q4 = pose.angle_rad - q2 - q3

        tolerance = 1e-9
        if not (
            g.joint4_min_rad - tolerance
            <= q4
            <= g.joint4_max_rad + tolerance
        ):
            raise ValueError(
                "Pose requires joint 4 outside its limits "
                f"[{math.degrees(g.joint4_min_rad):.0f}, "
                f"{math.degrees(g.joint4_max_rad):.0f}] degrees: "
                f"{math.degrees(q4):.2f} degrees"
            )
        return (q1, q2, q3, q4)

    def forward(self, joints: Sequence[float]) -> ArmPose:
        """Convert four joint angles in radians to an end-effector pose."""
        q1, q2, q3, q4 = _first_four(joints, "joints")
        g = self.geometry
        q23 = q2 + q3
        pitch = q23 + q4
        radius = (
            g.a2 * math.cos(q2)
            + g.a3 * math.cos(q23)
            + g.a4 * math.cos(pitch)
        )
        z = (
            g.l1
            + g.a2 * math.sin(q2)
            + g.a3 * math.sin(q23)
            + g.a4 * math.sin(pitch)
        )
        return ArmPose(
            x=radius * math.cos(q1),
            y=radius * math.sin(q1),
            z=z,
            angle_rad=pitch,
        )

    def remove_offsets(self, joints: Sequence[float]) -> tuple[float, ...]:
        values = _first_four(joints, "joints")
        return tuple(
            angle - offset
            for angle, offset in zip(values, self.geometry.angle_offsets)
        )

    def add_offsets(self, joints: Sequence[float]) -> tuple[float, ...]:
        values = _first_four(joints, "joints")
        return tuple(
            angle + offset
            for angle, offset in zip(values, self.geometry.angle_offsets)
        )


def rad_to_dxl(radians: Sequence[float]) -> list[int]:
    """Convert joint angles in radians to raw Dynamixel position values."""
    return [
        round((-math.degrees(angle) + 90.0) / 180.0 * 2048.0 + 1024.0)
        for angle in radians
    ]


def dxl_to_rad(positions: Sequence[int]) -> list[float]:
    """Convert the first four raw Dynamixel positions to radians."""
    values = _first_four(positions, "positions")
    return [
        -math.radians((float(position) - 1024.0) / 2048.0 * 180.0 - 90.0)
        for position in values
    ]


def synchronized_joint_speeds(
    start: Sequence[float],
    target: Sequence[float],
    maximum_speed: int,
    minimum_speed: int = 10,
) -> list[int]:
    """Scale speeds so all four joints approximately arrive together."""
    start_values = _first_four(start, "start")
    target_values = _first_four(target, "target")
    maximum_speed = max(1, int(maximum_speed))
    minimum_speed = max(1, min(maximum_speed, int(minimum_speed)))

    changes = [
        abs(float(end) - float(begin))
        for begin, end in zip(start_values, target_values)
    ]
    largest_change = max(changes)
    if largest_change == 0.0:
        return [minimum_speed] * 4
    return [
        maximum_speed
        if math.isclose(change, largest_change)
        else max(minimum_speed, round(maximum_speed * change / largest_change))
        for change in changes
    ]


def interpolate_joint_steps(
    start: Sequence[float],
    target: Sequence[float],
    max_step_deg: float,
) -> list[tuple[float, float, float, float]]:
    """Return joint-space waypoints with a bounded per-step angle change."""
    if not math.isfinite(max_step_deg) or max_step_deg <= 0.0:
        raise ValueError("max_step_deg must be a positive finite number")

    start_values = _first_four(start, "start")
    target_values = _first_four(target, "target")
    changes_deg = [
        abs(math.degrees(end - begin))
        for begin, end in zip(start_values, target_values)
    ]
    maximum_change = max(changes_deg)
    if maximum_change == 0.0:
        return []

    step_count = max(1, math.ceil(maximum_change / max_step_deg))
    return [
        tuple(
            begin + (index / step_count) * (end - begin)
            for begin, end in zip(start_values, target_values)
        )
        for index in range(1, step_count + 1)
    ]


def _first_four(values: Sequence[float], name: str) -> tuple:
    if len(values) < 4:
        raise ValueError(f"{name} must contain at least four values")
    return tuple(values[:4])
