from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ArmPose:
    x: float
    y: float
    z: float
    angle_rad: float


@dataclass(frozen=True)
class ArmGeometry:
    l1: float = 0.080 # base offset height
    a2: float = 0.130 # first link length
    a3: float = 0.124 # second link length
    #a4: float = 0.146 # wrist/tool offset
    a4: float = 0.111 # third link length
    joint4_min_rad: float = math.radians(-110.0)
    joint4_max_rad: float = math.radians(90.0)

    @property
    def phi(self) -> float:
        return -math.atan2(0.024, 0.128)

    @property
    def angle_offset(self) -> list[float]:
        return [0.0, math.pi / 2 + self.phi, -(math.pi / 2 + self.phi), 0.0]


class ArmKinematics:
    def __init__(self, geometry: ArmGeometry | None = None) -> None:
        self.geometry = geometry or ArmGeometry()

    def inverse(self, pose: ArmPose) -> list[float]:
        '''
        Converts end-effector pose into joint angles in radians.
        ArmPose: x, y, z, pitch angle
        '''
        g = self.geometry
        q1 = math.atan2(pose.y, pose.x)
        r = math.hypot(pose.x, pose.y)
        z_rel = pose.z - g.l1
        r_c = r - g.a4 * math.cos(pose.angle_rad)
        z_c = z_rel - g.a4 * math.sin(pose.angle_rad)
        dist_sq = r_c * r_c + z_c * z_c
        cos_q3 = (dist_sq - g.a2**2 - g.a3**2) / (2 * g.a2 * g.a3)
        if cos_q3 < -1.0 or cos_q3 > 1.0:
            raise ValueError(f"pose is outside reachable workspace: {pose}")
        q3 = -math.acos(max(-1.0, min(1.0, cos_q3)))
        alpha = math.atan2(z_c, r_c)
        beta = math.atan2(g.a3 * math.sin(q3), g.a2 + g.a3 * math.cos(q3))
        q2 = alpha - beta
        q4 = pose.angle_rad - q2 - q3
        limit_tolerance = 1e-9
        if (
            q4 < g.joint4_min_rad - limit_tolerance
            or q4 > g.joint4_max_rad + limit_tolerance
        ):
            raise ValueError(
                "pose requires joint4 outside limits "
                f"[-110, 90] degrees: {math.degrees(q4):.2f} degrees"
            )
        return [q1, q2, q3, q4]

    def forward(self, joints: Sequence[float]) -> ArmPose:
        '''
        Converts joint angles into end-effector pose.
        '''
        if len(joints) < 4:
            raise ValueError("joints must contain at least four angles")
        q1, q2, q3, q4 = joints[:4]
        g = self.geometry
        q23 = q2 + q3
        psi = q23 + q4
        radius = g.a2 * math.cos(q2) + g.a3 * math.cos(q23) + g.a4 * math.cos(psi)
        z = g.l1 + g.a2 * math.sin(q2) + g.a3 * math.sin(q23) + g.a4 * math.sin(psi)
        return ArmPose(
            x=radius * math.cos(q1),
            y=radius * math.sin(q1),
            z=z,
            angle_rad=psi,
        )

    def remove_offsets(self, joints: Sequence[float]) -> list[float]:
        return [angle - offset for angle, offset in zip(joints, self.geometry.angle_offset)]

    def add_offsets(self, joints: Sequence[float]) -> list[float]:
        return [angle + offset for angle, offset in zip(joints, self.geometry.angle_offset)]


def rad_to_dxl(rad: Sequence[float]) -> list[int]:
    '''
    Converts joint angles in radians to Dynamixel position values.
    '''
    return [round((-math.degrees(angle) + 90) / 180 * 2048 + 1024) for angle in rad]


def dxl_to_rad(dxl: Sequence[int]) -> list[float]:
    return [-math.radians((float(pos) - 1024) / 2048 * 180 - 90) for pos in dxl[:4]]


def synchronized_joint_speeds(
    start: Sequence[float],
    target: Sequence[float],
    maximum_speed: int,
    minimum_speed: int = 10,
) -> list[int]:
    '''
    Scales per-join speeds so all 4 joints arrive at their target at the same time.
    The joint with the largest change will move at maximum_speed, 
    and the other joints will move at a speed proportional to their change.
    '''
    if len(start) < 4 or len(target) < 4:
        raise ValueError("start and target must contain at least four joint angles")
    maximum_speed = max(1, int(maximum_speed))
    minimum_speed = max(1, min(maximum_speed, int(minimum_speed)))
    changes = [abs(float(end) - float(begin)) for begin, end in zip(start[:4], target[:4])]
    largest_change = max(changes)
    if largest_change == 0.0:
        return [minimum_speed] * 4
    return [
        maximum_speed
        if change == largest_change
        else max(minimum_speed, round(maximum_speed * change / largest_change))
        for change in changes
    ]
