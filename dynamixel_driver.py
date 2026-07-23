from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from kinematics import ArmKinematics, dxl_to_rad, rad_to_dxl


@dataclass(frozen=True)
class DxlAddresses:
    op_mode: int = 11
    max_pos: int = 48
    min_pos: int = 52
    torque_enable: int = 64
    goal_current: int = 102
    profile_accel: int = 108
    profile_velocity: int = 112
    goal_pos: int = 116
    present_current: int = 126
    present_pos: int = 132


# Dynamixel X-series operating modes (control table address 11)
OP_MODE_CURRENT = 0
OP_MODE_POSITION = 3
OP_MODE_EXTENDED_POSITION = 4
OP_MODE_CURRENT_BASED_POSITION = 5


class DynamixelDriver:
    protocol_version = 2.0
    baudrate = 1_000_000
    min_dxl_position = 0
    max_dxl_position = 4095
    # The gripper runs in Extended Position Control mode (multi-turn), unlike
    # the single-turn 0-4095 arm joints -- its true mechanical closed/open
    # positions can legitimately fall outside [0, 4095] (e.g. just past the
    # 4095->0 wraparound). Clamping gripper commands to min/max_dxl_position
    # would silently truncate a valid target back to 4095 and it would never
    # reach the real hard stop. Wide enough for a couple of extra turns either
    # way without being fully unbounded.
    gripper_min_position = -4096
    gripper_max_position = 8192
    # Current register units are ~2.69 mA/tick on XL430/XM430-class motors.
    # Confirm against your gripper motor's control table before relying on
    # absolute mA values.
    current_ma_per_unit = 2.69

    def __init__(
        self,
        motor_ids: Sequence[int] = (11, 12, 13, 14, 15),
        default_speed: int = 100,
        default_accel: int = 20,
        io_retry_count: int = 2,
        io_retry_delay_sec: float = 0.03,
        goal_write_ack: bool = True,
    ) -> None:
        if len(motor_ids) != 5:
            raise ValueError("motor_ids must contain four joint ids and one gripper id")
        self.motor_ids = tuple(int(v) for v in motor_ids)
        self.default_speed = int(default_speed)
        self.default_accel = int(default_accel)
        self.io_retry_count = max(0, int(io_retry_count))
        self.io_retry_delay_sec = max(0.0, float(io_retry_delay_sec))
        self.goal_write_ack = bool(goal_write_ack)
        self.pos_limits = [(self.min_dxl_position, self.max_dxl_position)] * 5
        self.address = DxlAddresses()
        self.kinematics = ArmKinematics()
        self.port = None
        self.packet = None
        self.current_port = ""
        self.connected = False
        self.last_sent_dxl = [2048, 2048, 2048, 2048]
        self.io_lock = threading.RLock()
        self.io_lock_timeout_sec = 2.0
        self.gripper_op_mode = OP_MODE_EXTENDED_POSITION

    def connect(self, port_name: str) -> None:
        with self._locked_io("connect"):
            if self.connected or self.port is not None:
                self._disconnect_unlocked()
            if not port_name:
                raise RuntimeError("serial port is empty")
            if not os.path.exists(port_name):
                raise RuntimeError(f"serial device does not exist: {port_name}")
            if not os.access(port_name, os.R_OK | os.W_OK):
                raise RuntimeError(f"no read/write permission for {port_name}")
            try:
                from dynamixel_sdk import PacketHandler, PortHandler
            except ImportError as exc:
                raise RuntimeError(
                    "Python package 'dynamixel-sdk' is required. Install it with: pip install dynamixel-sdk"
                ) from exc
            port = PortHandler(port_name)
            packet = PacketHandler(self.protocol_version)
            if not port.openPort():
                raise RuntimeError(f"could not open serial port {port_name}")
            if not port.setBaudRate(self.baudrate):
                port.closePort()
                raise RuntimeError(f"could not set baudrate to {self.baudrate}")
            self.port = port
            self.packet = packet
            self.current_port = port_name
            self.connected = True
            try:
                self._init_motors_unlocked()
            except Exception:
                try:
                    self._disconnect_unlocked()
                except Exception:
                    pass
                raise

    def disconnect(self) -> None:
        with self._locked_io("disconnect"):
            self._disconnect_unlocked()

    def _disconnect_unlocked(self) -> None:
        torque_error: Exception | None = None
        close_error: Exception | None = None
        try:
            if self.connected and self.port is not None and self.packet is not None:
                for dxl_id in self.motor_ids:
                    try:
                        self._write1(dxl_id, self.address.torque_enable, 0, "disconnect disable torque")
                    except Exception as exc:
                        if torque_error is None:
                            torque_error = exc
        finally:
            if self.port is not None:
                try:
                    self.port.closePort()
                except Exception as exc:
                    close_error = exc
            self.port = None
            self.packet = None
            self.current_port = ""
            self.connected = False
        if close_error is not None:
            raise RuntimeError(f"could not close Dynamixel port cleanly: {close_error}") from close_error
        if torque_error is not None:
            raise RuntimeError(f"port closed, but torque disable failed: {torque_error}") from torque_error

    def init_motors(self) -> None:
        with self._locked_io("initialize motors"):
            self._init_motors_unlocked()

    def _init_motors_unlocked(self) -> None:
        self._require_connection()
        self.gripper_op_mode = OP_MODE_EXTENDED_POSITION
        for idx, dxl_id in enumerate(self.motor_ids):
            motor_role = "gripper" if idx == 4 else f"joint {idx + 1}"
            self._write1(dxl_id, self.address.torque_enable, 0, f"init {motor_role}")
            self._write1(dxl_id, self.address.op_mode, self.gripper_op_mode if idx == 4 else 3, f"init {motor_role}")
            if idx < 4:
                min_pos, max_pos = self.pos_limits[idx]
                self._write4(dxl_id, self.address.min_pos, min_pos, f"init {motor_role}")
                self._write4(dxl_id, self.address.max_pos, max_pos, f"init {motor_role}")
            self._write4(dxl_id, self.address.profile_velocity, self.default_speed, f"init {motor_role}")
            self._write4(dxl_id, self.address.profile_accel, self.default_accel, f"init {motor_role}")
            self._write1(dxl_id, self.address.torque_enable, 1, f"init {motor_role}")

    def set_joint_angles(self, joints: Sequence[float]) -> None:
        with self._locked_io("set joint angles"):
            if len(joints) < 4:
                raise ValueError("joints must contain at least four angles")
            dxl_pos = rad_to_dxl(self.kinematics.remove_offsets(joints[:4]))
            self.last_sent_dxl = [self._clamp_int(pos, *limit) for pos, limit in zip(dxl_pos, self.pos_limits[:4])]
            if not self.connected:
                return
            for idx, (dxl_id, pos) in enumerate(zip(self.motor_ids[:4], self.last_sent_dxl)):
                self._write_goal_position(dxl_id, pos, f"set joint angles joint {idx + 1}")

    def set_joint_speeds(self, speeds: Sequence[int]) -> None:
        with self._locked_io("set joint speeds"):
            if len(speeds) < 4:
                raise ValueError("speeds must contain at least four values")
            joint_speeds = [max(1, int(speed)) for speed in speeds[:4]]
            if not self.connected:
                return
            for idx, (dxl_id, speed) in enumerate(zip(self.motor_ids[:4], joint_speeds)):
                self._write4(
                    dxl_id,
                    self.address.profile_velocity,
                    speed,
                    f"set joint speed joint {idx + 1}",
                )

    def read_joint_angles(self) -> list[float]:
        with self._locked_io("read joint angles"):
            if not self.connected:
                return self.kinematics.add_offsets(dxl_to_rad(self.last_sent_dxl))
            dxl_positions = [
                self._to_signed32(self._read4(dxl_id, self.address.present_pos, f"read joint {idx + 1}"))
                for idx, dxl_id in enumerate(self.motor_ids[:4])
            ]
            return self.kinematics.add_offsets(dxl_to_rad(dxl_positions))

    def set_gripper(self, position: int) -> int:
        '''Plain position command; kept for backward compatibility. Uses whichever
        operating mode the gripper is currently in (see set_gripper_position/
        set_gripper_current to control that explicitly).'''
        with self._locked_io("set gripper"):
            position = self._clamp_int(position, self.gripper_min_position, self.gripper_max_position)
            if self.connected:
                self._write_goal_position(self.motor_ids[4], position, "set gripper")
            return position

    def set_gripper_position(self, position: int, max_current: int | None = None) -> int:
        '''
        Position-demand gripper control.
        If max_current is given (raw current-register units), switches the gripper to
        Current-based Position Control so it drives toward `position` but stalls
        safely once it draws that much current -- e.g. closing on an object without
        crushing it. If max_current is None, uses plain Position Control at full torque.
        '''
        with self._locked_io("set gripper position"):
            position = self._clamp_int(position, self.gripper_min_position, self.gripper_max_position)
            target_mode = OP_MODE_CURRENT_BASED_POSITION if max_current is not None else OP_MODE_EXTENDED_POSITION
            self._set_gripper_operating_mode_unlocked(target_mode)
            if not self.connected:
                return position
            if max_current is not None:
                current = self._clamp_int(max_current, 0, 32767)
                self._write2(self.motor_ids[4], self.address.goal_current, current, "set gripper max current")
            self._write_goal_position(self.motor_ids[4], position, "set gripper position")
            return position

    def set_gripper_current(self, current: int) -> int:
        '''
        Current-demand (force) gripper control. Switches to pure Current Control
        and drives the gripper closed/open at up to `current` (raw units, ~2.69 mA
        each on X-series motors) with no position target -- useful for "squeeze at
        force X" grasps where the exact finger position doesn't matter.
        Positive values close the gripper, negative values open it (direction
        depends on motor orientation -- verify on your hardware before relying on sign).
        '''
        with self._locked_io("set gripper current"):
            current = self._clamp_int(current, -32768, 32767)
            self._set_gripper_operating_mode_unlocked(OP_MODE_CURRENT)
            if self.connected:
                self._write2(self.motor_ids[4], self.address.goal_current, current, "set gripper current")
            return current

    def read_gripper_position(self) -> int:
        with self._locked_io("read gripper position"):
            if not self.connected:
                return self.last_sent_dxl[-1] if len(self.last_sent_dxl) > 4 else 2048
            return self._to_signed32(
                self._read4(self.motor_ids[4], self.address.present_pos, "read gripper position")
            )

    def read_gripper_current_ma(self) -> float:
        '''Present current draw on the gripper motor, in approximate mA (see current_ma_per_unit).'''
        with self._locked_io("read gripper current"):
            if not self.connected:
                return 0.0
            raw = self._to_signed16(
                self._read2(self.motor_ids[4], self.address.present_current, "read gripper current")
            )
            return raw * self.current_ma_per_unit

    def set_gripper_speed(self, speed: int) -> None:
        '''Set the gripper's profile velocity (max speed) independently of the
        arm joints -- set_joint_speeds() only covers motor_ids[:4]. Lower values
        close/open more slowly and gently; higher values move faster.'''
        with self._locked_io("set gripper speed"):
            speed = max(1, int(speed))
            if not self.connected:
                return
            self._write4(self.motor_ids[4], self.address.profile_velocity, speed, "set gripper speed")

    def set_gripper_torque(self, enabled: bool) -> None:
        '''Enable/disable torque on the gripper motor only, leaving the four arm
        joints untouched (unlike disable_torque(), which affects all motors).
        Useful for hand-moving the gripper to its mechanical limits during calibration.'''
        with self._locked_io("set gripper torque"):
            if not self.connected:
                return
            self._write1(self.motor_ids[4], self.address.torque_enable, 1 if enabled else 0, "set gripper torque")

    def _set_gripper_operating_mode_unlocked(self, mode: int) -> None:
        if self.gripper_op_mode == mode:
            return
        self.gripper_op_mode = mode
        if not self.connected:
            return
        dxl_id = self.motor_ids[4]
        self._write1(dxl_id, self.address.torque_enable, 0, "set gripper op mode (disable torque)")
        self._write1(dxl_id, self.address.op_mode, mode, "set gripper op mode")
        self._write1(dxl_id, self.address.torque_enable, 1, "set gripper op mode (enable torque)")

    def disable_torque(self) -> None:
        with self._locked_io("disable torque"):
            if not self.connected:
                return
            for dxl_id in self.motor_ids:
                self._write1(dxl_id, self.address.torque_enable, 0, "disable torque")

    @contextmanager
    def _locked_io(self, operation: str) -> Iterator[None]:
        acquired = self.io_lock.acquire(timeout=self.io_lock_timeout_sec)
        if not acquired:
            port = self.current_port or "the Dynamixel port"
            raise RuntimeError(
                f"Dynamixel serial bus is busy while trying to {operation} on {port}. "
                "This usually means another arm operation is still using the SDK port; "
                "if it persists, check that old control_node or a second hardware_arm_node is not running."
            )
        try:
            yield
        finally:
            self.io_lock.release()

    def _write1(self, dxl_id: int, address: int, value: int, operation: str = "write1") -> None:
        self._txrx(
            lambda: self.packet.write1ByteTxRx(self.port, dxl_id, address, int(value)),
            self._describe_io(operation, "write1", dxl_id, address, value),
        )

    def _write2(self, dxl_id: int, address: int, value: int, operation: str = "write2") -> None:
        self._txrx(
            lambda: self.packet.write2ByteTxRx(self.port, dxl_id, address, int(value) & 0xFFFF),
            self._describe_io(operation, "write2", dxl_id, address, value),
        )

    def _write4(self, dxl_id: int, address: int, value: int, operation: str = "write4") -> None:
        self._txrx(
            lambda: self.packet.write4ByteTxRx(self.port, dxl_id, address, int(value)),
            self._describe_io(operation, "write4", dxl_id, address, value),
        )

    def _write_goal_position(self, dxl_id: int, value: int, operation: str) -> None:
        if self.goal_write_ack:
            self._write4(dxl_id, self.address.goal_pos, value, operation)
            return
        tx_only = getattr(self.packet, "write4ByteTxOnly", None)
        if not callable(tx_only):
            self._write4(dxl_id, self.address.goal_pos, value, operation)
            return
        self._txonly(
            lambda: tx_only(self.port, dxl_id, self.address.goal_pos, int(value)),
            self._describe_io(operation, "write4_tx_only", dxl_id, self.address.goal_pos, value),
        )

    def _read2(self, dxl_id: int, address: int, operation: str = "read2") -> int:
        value = self._txrx(
            lambda: self.packet.read2ByteTxRx(self.port, dxl_id, address),
            self._describe_io(operation, "read2", dxl_id, address),
        )
        return int(value)

    def _read4(self, dxl_id: int, address: int, operation: str = "read4") -> int:
        value = self._txrx(
            lambda: self.packet.read4ByteTxRx(self.port, dxl_id, address),
            self._describe_io(operation, "read4", dxl_id, address),
        )
        return int(value)

    def _txrx(self, call: Callable[[], Sequence[int]], description: str) -> int:
        attempts = self.io_retry_count + 1
        for attempt in range(attempts):
            try:
                response = call()
                if len(response) == 3:
                    value, result, error = response
                else:
                    result, error = response
                    value = 0
                self._check_result(int(result), int(error))
                return int(value)
            except RuntimeError as exc:
                if attempt + 1 >= attempts or not self._is_retryable_txrx_error(str(exc)):
                    suffix = f" after {attempt + 1} attempts" if attempt else ""
                    raise RuntimeError(f"{description} failed{suffix}: {exc}") from exc
                self._recover_after_txrx_error()
                if self.io_retry_delay_sec > 0:
                    time.sleep(self.io_retry_delay_sec)
        raise RuntimeError(f"{description} failed: Dynamixel transaction did not complete")

    def _txonly(self, call: Callable[[], int | Sequence[int]], description: str) -> None:
        result = call()
        if isinstance(result, Sequence) and not isinstance(result, (bytes, bytearray, str)):
            if len(result) == 0:
                return
            result_code = int(result[0])
        else:
            result_code = int(result)
        self._check_result(result_code, 0)

    def _check_result(self, result: int, error: int) -> None:
        if result != 0:
            message = self.packet.getTxRxResult(result)
            if "Port is in use" in message:
                message = (
                    f"{message}. Dynamixel SDK reported a busy serial transaction; "
                    "check that only one arm node owns the port and that hardware access is serialized."
                )
            raise RuntimeError(message)
        if error:
            raise RuntimeError(self.packet.getRxPacketError(error))

    def _recover_after_txrx_error(self) -> None:
        if self.port is None:
            return
        clear_port = getattr(self.port, "clearPort", None)
        if callable(clear_port):
            try:
                clear_port()
            except Exception:
                pass

    @staticmethod
    def _is_retryable_txrx_error(message: str) -> bool:
        text = message.lower()
        return "no status packet" in text or "incorrect status packet" in text or "timeout" in text

    def _describe_io(
        self,
        operation: str,
        command: str,
        dxl_id: int,
        address: int,
        value: int | None = None,
    ) -> str:
        parts = [
            str(operation),
            command,
            f"id={int(dxl_id)}",
            f"addr={int(address)}({self._address_name(address)})",
        ]
        if value is not None:
            parts.append(f"value={int(value)}")
        return " ".join(parts)

    def _address_name(self, address: int) -> str:
        for name, value in self.address.__dict__.items():
            if value == address:
                return name
        return "unknown"

    def _require_connection(self) -> None:
        if not self.connected:
            raise RuntimeError("Dynamixel bus is not connected")

    def close_port_keep_torque(self) -> None:
        """
        Close the serial port without explicitly disabling motor torque.
        Use this if you want the motors to hold their last commanded position
        after the controller exits. This will *not* send torque-disable
        commands to the motors; ensure this is what you intend for safety.
        """
        with self._locked_io("close port (keep torque enabled)"):
            # Close the underlying PortHandler if present, but do not write
            # any torque-disable commands. Leave motor torque state unchanged.
            if self.port is not None:
                try:
                    close_port = getattr(self.port, "closePort", None)
                    if callable(close_port):
                        close_port()
                finally:
                    self.port = None
                    self.packet = None
                    self.current_port = ""
                    self.connected = False

    @staticmethod
    def _to_signed32(value: int) -> int:
        return value - 2**32 if value >= 2**31 else value

    @staticmethod
    def _to_signed16(value: int) -> int:
        return value - 2**16 if value >= 2**15 else value

    @staticmethod
    def _clamp_int(value: float, low: int, high: int) -> int:
        return max(low, min(high, round(float(value))))
