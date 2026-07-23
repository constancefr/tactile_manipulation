"""High-level orchestration for one tactile key reorientation cycle.

A key with a detected embossed line is rotated so the line is vertical, then
placed gently on its flat base. A key without the feature is discarded to the
left using a faster sweeping motion.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol, Sequence

import cv2
import numpy as np

from .key_classifier import (
    EmbossedFeatureClassifier,
    KeyClassification,
    KeyLabel,
)
from .kinematics import (
    ArmPose,
    interpolate_joint_steps,
    synchronized_joint_speeds,
)
from .robot_hardware import RobotHardware
from .tactile_detector import DetectionResult, TactileBandDetector


class TactileCameraProtocol(Protocol):
    def capture_frame(self) -> np.ndarray: ...


@dataclass(frozen=True)
class ReorientationPoses:
    """Fixed cell poses required by the reorientation task."""

    home: ArmPose
    grasp: ArmPose
    lift: ArmPose

    # Defect path: wind up on the right, sweep left, then release.
    defect_windup: ArmPose
    defect_release: ArmPose

    # Good-key path: rotate above this location, then lower to the table.
    good_above: ArmPose
    good_place: ArmPose

    def named(self) -> dict[str, ArmPose]:
        return {
            "home": self.home,
            "grasp": self.grasp,
            "lift": self.lift,
            "defect_windup": self.defect_windup,
            "defect_release": self.defect_release,
            "good_above": self.good_above,
            "good_place": self.good_place,
        }


@dataclass(frozen=True)
class OrientationCalibration:
    """Map DIGIT image angles to a physical wrist command.

    ``image_vertical_deg`` is the angle corresponding to gravity/vertical in
    the DIGIT image. For a normal OpenCV image this is usually 90 degrees.

    Set ``wrist_direction`` to +1 or -1 after checking whether a positive
    image correction requires a positive or negative joint-4 rotation.
    """

    image_vertical_deg: float = 90.0
    wrist_direction: int = 1
    wrist_zero_offset_deg: float = 0.0
    max_abs_wrist_command_deg: float = 90.0

    def __post_init__(self) -> None:
        if self.wrist_direction not in (-1, 1):
            raise ValueError("wrist_direction must be +1 or -1")
        values = (
            self.image_vertical_deg,
            self.wrist_zero_offset_deg,
            self.max_abs_wrist_command_deg,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("orientation calibration values must be finite")
        if self.max_abs_wrist_command_deg <= 0.0:
            raise ValueError("max_abs_wrist_command_deg must be positive")


@dataclass(frozen=True)
class ReorientationMotionConfig:
    """Special motion settings for rotation, placement, and discard."""

    wrist_speed: int = 30
    wrist_step_deg: float = 1.0
    wrist_step_delay_sec: float = 0.02

    placement_speed: int = 20
    placement_step_deg: float = 1.0
    placement_step_delay_sec: float = 0.04

    throw_speed: int = 100
    throw_step_deg: float = 4.0
    throw_step_delay_sec: float = 0.01

    def __post_init__(self) -> None:
        for name, value in (
            ("wrist_speed", self.wrist_speed),
            ("placement_speed", self.placement_speed),
            ("throw_speed", self.throw_speed),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        for name, value in (
            ("wrist_step_deg", self.wrist_step_deg),
            ("placement_step_deg", self.placement_step_deg),
            ("throw_step_deg", self.throw_step_deg),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        for name, value in (
            ("wrist_step_delay_sec", self.wrist_step_delay_sec),
            ("placement_step_delay_sec", self.placement_step_delay_sec),
            ("throw_step_delay_sec", self.throw_step_delay_sec),
        ):
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class ReorientationRunResult:
    classification: KeyClassification
    detection: DetectionResult
    measured_line_angle_deg: float | None
    image_correction_deg: float | None
    wrist_command_deg: float | None
    raw_image_path: Path
    annotated_image_path: Path
    preprocessed_image_path: Path


class KeyReorientationTask:
    """Perform one grasp, tactile inspection, and reorientation cycle."""

    WRIST_JOINT_INDEX = 3

    def __init__(
        self,
        robot: RobotHardware,
        camera: TactileCameraProtocol,
        detector: TactileBandDetector,
        classifier: EmbossedFeatureClassifier,
        poses: ReorientationPoses,
        *,
        orientation: OrientationCalibration | None = None,
        motion: ReorientationMotionConfig | None = None,
        output_dir: Path | str = "reorientation_results",
        gripper_max_current: int | None = 60,
        tactile_settle_delay_sec: float = 0.5,
        return_home: bool = True,
        blank_image: np.ndarray | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        if gripper_max_current is not None and gripper_max_current <= 0:
            raise ValueError("gripper_max_current must be positive")
        if tactile_settle_delay_sec < 0.0:
            raise ValueError("tactile_settle_delay_sec cannot be negative")
        if blank_image is not None and (
            not isinstance(blank_image, np.ndarray) or blank_image.ndim != 3
        ):
            raise ValueError("blank_image must be a BGR NumPy image")

        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.classifier = classifier
        self.poses = poses
        self.orientation = orientation or OrientationCalibration()
        self.motion = motion or ReorientationMotionConfig()
        self.output_dir = Path(output_dir)
        self.gripper_max_current = gripper_max_current
        self.tactile_settle_delay_sec = tactile_settle_delay_sec
        self.return_home = return_home
        self.blank_image = blank_image
        self._sleep = sleep
        self._log = log
        self._input = input_fn

    def validate_poses(self) -> None:
        """Validate every fixed pose before any hardware movement."""
        for name, pose in self.poses.named().items():
            self.robot.arm.solve_pose(pose)
            self._log(f"Pose validated: {name}")

    def run_once(self) -> ReorientationRunResult:
        """Execute one complete reorientation cycle.

        If inspection, angle estimation, or motion validation fails while the
        key is held, the gripper remains closed for operator recovery.
        """
        if not self.robot.connected:
            raise RuntimeError("RobotHardware must be connected before run_once()")

        holding_key = False
        released_key = False
        self.validate_poses()

        try:
            self._log("Moving to grasp pose")
            self.robot.arm.move_to_pose(self.poses.grasp)

            self._log("Opening gripper")
            self.robot.gripper.open(max_current=self.gripper_max_current)

            try:
                self._input(
                    "Place the key between the gripper fingers, clear your "
                    "hands, then press Enter..."
                )
            except (EOFError, OSError):
                # Allows hardware-free tests to capture stdin.
                pass

            self._log("Closing gripper around key")
            self.robot.gripper.close(max_current=self.gripper_max_current)
            holding_key = True

            if self.tactile_settle_delay_sec:
                self._sleep(self.tactile_settle_delay_sec)

            self._log("Capturing DIGIT tactile image")
            frame = self.camera.capture_frame()

            self._log("Detecting embossed-feature edges")
            detection = self.detector.detect(
                frame,
                blank_image=self.blank_image,
            )
            classification = self.classifier.classify(detection)

            measured_angle_deg: float | None = None
            image_correction_deg: float | None = None
            wrist_command_deg: float | None = None
            corrected_above: ArmPose | None = None
            corrected_place: ArmPose | None = None

            if classification.label is KeyLabel.GOOD:
                (
                    measured_angle_deg,
                    image_correction_deg,
                    wrist_command_deg,
                ) = self._estimate_rotation(detection)

                corrected_above = self._with_pitch_offset(
                    self.poses.good_above,
                    wrist_command_deg,
                )
                corrected_place = self._with_pitch_offset(
                    self.poses.good_place,
                    wrist_command_deg,
                )

                # Validate every runtime-dependent target before lifting.
                self.robot.arm.solve_pose(corrected_above)
                self.robot.arm.solve_pose(corrected_place)
                good_above_joints = self.robot.arm.solve_pose(
                    self.poses.good_above
                )
                self._validate_wrist_command(
                    wrist_command_deg,
                    current_joints=good_above_joints,
                )

                self._log(
                    "Classification: good; "
                    f"line={measured_angle_deg:.2f} deg, "
                    f"image correction={image_correction_deg:+.2f} deg, "
                    f"wrist command={wrist_command_deg:+.2f} deg"
                )
            else:
                self._log(
                    "Classification: defect "
                    f"({classification.edge_count} detected edges)"
                )

            paths = self._save_inspection_artifacts(
                frame=frame,
                detection=detection,
                classification=classification,
                measured_angle_deg=measured_angle_deg,
                image_correction_deg=image_correction_deg,
                wrist_command_deg=wrist_command_deg,
            )

            self._log("Lifting key clear of the grasp area")
            self.robot.arm.move_to_pose(self.poses.lift)

            if classification.label is KeyLabel.DEFECT:
                self._discard_defect()
                holding_key = False
                released_key = True
            else:
                if (
                    wrist_command_deg is None
                    or corrected_above is None
                    or corrected_place is None
                ):
                    raise RuntimeError("Good key has no validated rotation result")

                self._place_good_key(
                    wrist_command_deg=wrist_command_deg,
                    corrected_above=corrected_above,
                    corrected_place=corrected_place,
                )
                holding_key = False
                released_key = True

                self._log("Retreating without disturbing the upright key")
                self._move_to_pose_at_speed(
                    corrected_above,
                    maximum_speed=self.motion.placement_speed,
                    max_step_deg=self.motion.placement_step_deg,
                    step_delay_sec=self.motion.placement_step_delay_sec,
                )

            if self.return_home:
                self._log("Returning arm home")
                self.robot.arm.move_to_pose(self.poses.home)

            return ReorientationRunResult(
                classification=classification,
                detection=detection,
                measured_line_angle_deg=measured_angle_deg,
                image_correction_deg=image_correction_deg,
                wrist_command_deg=wrist_command_deg,
                raw_image_path=paths[0],
                annotated_image_path=paths[1],
                preprocessed_image_path=paths[2],
            )
        except Exception:
            if holding_key:
                self._log(
                    "Task failed while holding the key; leaving the gripper "
                    "closed for operator recovery."
                )
            elif released_key:
                self._log("Task failed after the key had already been released.")
            raise

    def _discard_defect(self) -> None:
        """Sweep left quickly and release the defective key."""
        self._log("Moving to defect wind-up pose")
        self.robot.arm.move_to_pose(self.poses.defect_windup)

        self._log("Sweeping left to discard defect")
        self._move_to_pose_at_speed(
            self.poses.defect_release,
            maximum_speed=self.motion.throw_speed,
            max_step_deg=self.motion.throw_step_deg,
            step_delay_sec=self.motion.throw_step_delay_sec,
        )

        self._log("Releasing defective key")
        self.robot.gripper.open(max_current=self.gripper_max_current)

    def _place_good_key(
        self,
        *,
        wrist_command_deg: float,
        corrected_above: ArmPose,
        corrected_place: ArmPose,
    ) -> None:
        """Rotate the key, lower it onto its base, release, and retreat."""
        self._log("Moving above upright placement location")
        self.robot.arm.move_to_pose(self.poses.good_above)

        self._log(f"Rotating wrist by {wrist_command_deg:+.2f} degrees")
        self._rotate_wrist_by(wrist_command_deg)

        # Rotating joint 4 alone moves the tool tip through an arc. This
        # Cartesian move returns the tip to the intended x/y/z while retaining
        # the new pitch.
        self._log("Compensating tool-tip position after wrist rotation")
        self._move_to_pose_at_speed(
            corrected_above,
            maximum_speed=self.motion.wrist_speed,
            max_step_deg=self.motion.wrist_step_deg,
            step_delay_sec=self.motion.wrist_step_delay_sec,
        )

        self._log("Lowering key gently onto its flat base")
        self._move_to_pose_at_speed(
            corrected_place,
            maximum_speed=self.motion.placement_speed,
            max_step_deg=self.motion.placement_step_deg,
            step_delay_sec=self.motion.placement_step_delay_sec,
        )

        self._log("Opening gripper")
        self.robot.gripper.open(max_current=self.gripper_max_current)

    def _estimate_rotation(
        self,
        detection: DetectionResult,
    ) -> tuple[float, float, float]:
        """Return measured image angle, image correction, and wrist command."""
        theta = detection.dominant_angle_rad
        if theta is None:
            raise RuntimeError(
                "Key was classified as good, but no dominant line angle was found"
            )

        vertical = math.radians(self.orientation.image_vertical_deg)
        correction_rad = self._wrap_axial_angle(vertical - theta)
        measured_angle_deg = math.degrees(theta)
        image_correction_deg = math.degrees(correction_rad)
        wrist_command_deg = (
            self.orientation.wrist_direction * image_correction_deg
            + self.orientation.wrist_zero_offset_deg
        )

        if abs(wrist_command_deg) > self.orientation.max_abs_wrist_command_deg:
            raise RuntimeError(
                "Estimated wrist command exceeds the configured safety limit: "
                f"{wrist_command_deg:+.2f} deg; allowed "
                f"±{self.orientation.max_abs_wrist_command_deg:.2f} deg"
            )
        return measured_angle_deg, image_correction_deg, wrist_command_deg

    def _validate_wrist_command(
        self,
        degrees: float,
        *,
        current_joints: Sequence[float] | None = None,
    ) -> None:
        if not math.isfinite(degrees):
            raise ValueError("wrist command must be finite")

        current = (
            tuple(current_joints)
            if current_joints is not None
            else self.robot.arm.read_joint_angles()
        )
        if len(current) < 4:
            raise ValueError("current_joints must contain at least four values")
        target = current[self.WRIST_JOINT_INDEX] + math.radians(degrees)
        geometry = self.robot.arm.kinematics.geometry
        minimum = geometry.joint4_min_rad
        maximum = geometry.joint4_max_rad
        if not minimum <= target <= maximum:
            raise RuntimeError(
                "Requested wrist rotation exceeds joint-4 limits: "
                f"target={math.degrees(target):.2f} deg, "
                f"allowed=[{math.degrees(minimum):.2f}, "
                f"{math.degrees(maximum):.2f}] deg"
            )

    def _rotate_wrist_by(self, degrees: float) -> None:
        """Rotate only joint 4 using the arm controller's driver and limits."""
        self._validate_wrist_command(degrees)
        current = self.robot.arm.read_joint_angles()
        target = list(current)
        target[self.WRIST_JOINT_INDEX] += math.radians(degrees)
        self._move_joint_target(
            target,
            maximum_speed=self.motion.wrist_speed,
            max_step_deg=self.motion.wrist_step_deg,
            step_delay_sec=self.motion.wrist_step_delay_sec,
        )

    def _move_to_pose_at_speed(
        self,
        pose: ArmPose,
        *,
        maximum_speed: int,
        max_step_deg: float,
        step_delay_sec: float,
    ) -> None:
        target = self.robot.arm.solve_pose(pose)
        self._move_joint_target(
            target,
            maximum_speed=maximum_speed,
            max_step_deg=max_step_deg,
            step_delay_sec=step_delay_sec,
        )

    def _move_joint_target(
        self,
        target_joints: Sequence[float],
        *,
        maximum_speed: int,
        max_step_deg: float,
        step_delay_sec: float,
    ) -> None:
        """Move to a joint target with task-specific speed settings."""
        if len(target_joints) < 4:
            raise ValueError("target_joints must contain at least four values")
        if maximum_speed <= 0:
            raise ValueError("maximum_speed must be positive")
        if max_step_deg <= 0.0:
            raise ValueError("max_step_deg must be positive")
        if step_delay_sec < 0.0:
            raise ValueError("step_delay_sec cannot be negative")

        arm = self.robot.arm
        current = arm.read_joint_angles()
        waypoints = interpolate_joint_steps(
            current,
            target_joints[:4],
            max_step_deg,
        )
        minimum_speed = min(arm.motion.min_speed, maximum_speed)

        for waypoint in waypoints:
            speeds = synchronized_joint_speeds(
                current,
                waypoint,
                maximum_speed=maximum_speed,
                minimum_speed=minimum_speed,
            )
            arm.driver.set_joint_speeds(speeds)
            arm.driver.set_joint_angles(waypoint)
            if step_delay_sec:
                self._sleep(step_delay_sec)
            current = arm.read_joint_angles()

    def _save_inspection_artifacts(
        self,
        *,
        frame: np.ndarray,
        detection: DetectionResult,
        classification: KeyClassification,
        measured_angle_deg: float | None,
        image_correction_deg: float | None,
        wrist_command_deg: float | None,
    ) -> tuple[Path, Path, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.output_dir / f"{timestamp}_{classification.label.value}"
        run_dir.mkdir(parents=True, exist_ok=False)

        raw_path = run_dir / "digit_raw.png"
        annotated_path = run_dir / "digit_annotated.png"
        preprocessed_path = run_dir / "digit_preprocessed.png"

        annotated = self._add_orientation_overlay(
            detection.annotated_image,
            measured_angle_deg=measured_angle_deg,
            image_correction_deg=image_correction_deg,
            wrist_command_deg=wrist_command_deg,
        )

        self._write_image(raw_path, frame)
        self._write_image(annotated_path, annotated)
        self._write_image(preprocessed_path, detection.enhanced_image)

        metadata_lines = [
            f"label={classification.label.value}",
            f"edge_count={classification.edge_count}",
            f"minimum_good_edges={classification.minimum_good_edges}",
            f"measured_line_angle_deg={self._format_optional(measured_angle_deg)}",
            f"image_correction_deg={self._format_optional(image_correction_deg)}",
            f"wrist_command_deg={self._format_optional(wrist_command_deg)}",
        ]
        (run_dir / "result.txt").write_text(
            "\n".join(metadata_lines) + "\n",
            encoding="utf-8",
        )
        return raw_path, annotated_path, preprocessed_path

    def _add_orientation_overlay(
        self,
        image: np.ndarray,
        *,
        measured_angle_deg: float | None,
        image_correction_deg: float | None,
        wrist_command_deg: float | None,
    ) -> np.ndarray:
        result = image.copy()
        height, width = result.shape[:2]
        centre_x = width // 2

        # Reference direction corresponding to vertical in the image.
        cv2.line(
            result,
            (centre_x, 0),
            (centre_x, height - 1),
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        if measured_angle_deg is None:
            label = "No orientation used (defect)"
        else:
            label = (
                f"Line {measured_angle_deg:.1f} deg | "
                f"correction {image_correction_deg:+.1f} deg | "
                f"wrist {wrist_command_deg:+.1f} deg"
            )

        cv2.rectangle(result, (6, 48), (min(width - 1, 620), 84), (0, 0, 0), -1)
        cv2.putText(
            result,
            label,
            (14, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return result

    @staticmethod
    def _with_pitch_offset(pose: ArmPose, offset_deg: float) -> ArmPose:
        return ArmPose(
            x=pose.x,
            y=pose.y,
            z=pose.z,
            angle_rad=pose.angle_rad + math.radians(offset_deg),
        )

    @staticmethod
    def _wrap_axial_angle(angle_rad: float) -> float:
        """Wrap a line-orientation correction to [-90, 90) degrees."""
        return ((angle_rad + math.pi / 2.0) % math.pi) - math.pi / 2.0

    @staticmethod
    def _format_optional(value: float | None) -> str:
        return "none" if value is None else f"{value:.6f}"

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")
