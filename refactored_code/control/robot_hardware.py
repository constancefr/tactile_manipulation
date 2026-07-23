"""Shared connection lifecycle for arm and gripper controllers."""

from __future__ import annotations

from .arm_controller import ArmController, MotionConfig, Workspace
from .gripper_controller import GripperCalibration, GripperController
from .kinematics import ArmKinematics
from .protocols import DynamixelDriverProtocol


class RobotHardware:
    """Own one driver connection and expose arm/gripper controllers."""

    def __init__(
        self,
        driver: DynamixelDriverProtocol,
        port: str,
        gripper_calibration: GripperCalibration,
        *,
        kinematics: ArmKinematics | None = None,
        workspace: Workspace | None = None,
        motion: MotionConfig | None = None,
        keep_torque_on_close: bool = True,
    ) -> None:
        if not port:
            raise ValueError("port cannot be empty")
        self.driver = driver
        self.port = port
        self.keep_torque_on_close = keep_torque_on_close
        self.arm = ArmController(
            driver,
            kinematics=kinematics,
            workspace=workspace,
            motion=motion,
        )
        self.gripper = GripperController(driver, gripper_calibration)
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> "RobotHardware":
        if not self._connected:
            self.driver.connect(self.port)
            self._connected = True
        return self

    def close(self) -> None:
        if not self._connected:
            return
        try:
            if self.keep_torque_on_close:
                self.driver.close_port_keep_torque()
            else:
                self.driver.disconnect()
        finally:
            self._connected = False

    def __enter__(self) -> "RobotHardware":
        return self.connect()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
