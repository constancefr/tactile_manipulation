"""High-level orchestration for one tactile key pick-and-sort cycle."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from .key_classifier import (
    EmbossedFeatureClassifier,
    KeyClassification,
    KeyLabel,
)
from .kinematics import ArmPose
from .robot_hardware import RobotHardware
from .tactile_detector import DetectionResult, TactileBandDetector


class TactileCameraProtocol(Protocol):
    def capture_frame(self) -> np.ndarray: ...


@dataclass(frozen=True)
class SortingPoses:
    """All fixed poses required by the sorting cell."""

    home: ArmPose
    pick_approach: ArmPose
    pick_grasp: ArmPose
    pick_lift: ArmPose
    good_approach: ArmPose
    good_drop: ArmPose
    defect_approach: ArmPose
    defect_drop: ArmPose

    def destination_for(self, label: KeyLabel) -> tuple[ArmPose, ArmPose]:
        if label is KeyLabel.GOOD:
            return self.good_drop
        return self.defect_drop
    # def destination_for(self, classification_output) -> tuple[ArmPose, ArmPose]:
    #     if classification_output is not None:
    #         return self.good_approach, self.good_drop
    #     return self.defect_approach, self.defect_drop

    def named(self) -> dict[str, ArmPose]:
        return {
            "home": self.home,
            "pick_approach": self.pick_approach,
            "pick_grasp": self.pick_grasp,
            "pick_lift": self.pick_lift,
            "good_approach": self.good_approach,
            "good_drop": self.good_drop,
            "defect_approach": self.defect_approach,
            "defect_drop": self.defect_drop,
        }


@dataclass(frozen=True)
class SortRunResult:
    classification: KeyClassification
    detection: DetectionResult
    raw_image_path: Path
    annotated_image_path: Path
    preprocessed_image_path: Path


class KeySortingTask:
    """Perform one complete grasp, tactile inspection, and sorting cycle."""

    def __init__(
        self,
        robot: RobotHardware,
        camera: TactileCameraProtocol,
        detector: TactileBandDetector,
        classifier: EmbossedFeatureClassifier,
        poses: SortingPoses,
        *,
        output_dir: Path | str = "sorting_results",
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
        self.output_dir = Path(output_dir)
        self.gripper_max_current = gripper_max_current
        self.tactile_settle_delay_sec = tactile_settle_delay_sec
        self.return_home = return_home
        self.blank_image = blank_image
        self._sleep = sleep
        self._log = log
        self._input = input_fn

    def validate_poses(self) -> None:
        """Run inverse kinematics for every fixed pose before moving hardware."""
        for name, pose in self.poses.named().items():
            self.robot.arm.solve_pose(pose)
            self._log(f"Pose validated: {name}")

    def run_once(self) -> SortRunResult:
        """Execute one complete key-sorting cycle.

        If sensing or classification fails after grasping, the gripper remains
        closed. The object is not sent to either bucket without a valid label.
        """
        if not self.robot.connected:
            raise RuntimeError("RobotHardware must be connected before run_once()")

        holding_key = False
        released_key = False
        self.validate_poses()

        try:
            self._log("Opening gripper")
            self.robot.gripper.open(max_current=self.gripper_max_current)

            # self._log("Moving above key")
            # self.robot.arm.move_to_pose(self.poses.pick_approach)

            # Wait for user to place the key in the gripper's grasping area
            try:
                self._input("Press Enter when ready to grasp the key...")
            except (EOFError, OSError):
                # Non-interactive environments such as pytest capture stdin.
                # Skip the pause there so the task remains testable.
                pass

            self._log("Descending to grasp pose")
            self.robot.arm.move_to_pose(self.poses.pick_grasp)

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
            self._log(
                f"Classification: {classification.label.value} "
                f"({classification.edge_count} detected edges)"
            )

            paths = self._save_inspection_artifacts(
                frame,
                detection,
                classification,
            )

            # self._log("Lifting key clear of stand")
            # self.robot.arm.move_to_pose(self.poses.pick_lift)

            drop_pose = self.poses.destination_for(
                classification.label
            )
            self._log(
                f"Moving to {classification.label.value} bucket approach"
            )
            # self.robot.arm.move_to_pose(approach_pose)

            self._log("Lowering key")
            self.robot.arm.move_to_pose(drop_pose)

            self._log("Releasing key")
            self.robot.gripper.open(max_current=self.gripper_max_current)
            holding_key = False
            released_key = True

            # self._log("Retreating from bucket")
            # self.robot.arm.move_to_pose(approach_pose)

            if self.return_home:
                self._log("Returning arm home")
                self.robot.arm.move_to_pose(self.poses.home)

            return SortRunResult(
                classification=classification,
                detection=detection,
                raw_image_path=paths[0],
                annotated_image_path=paths[1],
                preprocessed_image_path=paths[2],
            )
        except Exception:
            if holding_key:
                self._log(
                    "Task failed while holding the key; leaving the gripper "
                    "closed and not choosing a bucket."
                )
            elif released_key:
                self._log("Task failed after the key had already been released.")
            raise

    def _save_inspection_artifacts(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        classification: KeyClassification,
    ) -> tuple[Path, Path, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.output_dir / f"{timestamp}_{classification.label.value}"
        run_dir.mkdir(parents=True, exist_ok=False)

        raw_path = run_dir / "digit_raw.png"
        annotated_path = run_dir / "digit_annotated.png"
        preprocessed_path = run_dir / "digit_preprocessed.png"

        self._write_image(raw_path, frame)
        self._write_image(annotated_path, detection.annotated_image)
        self._write_image(preprocessed_path, detection.enhanced_image)

        metadata = run_dir / "result.txt"
        metadata.write_text(
            "\n".join(
                [
                    f"label={classification.label.value}",
                    f"edge_count={classification.edge_count}",
                    f"minimum_good_edges={classification.minimum_good_edges}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return raw_path, annotated_path, preprocessed_path

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")
