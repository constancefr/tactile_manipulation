"""High-level orchestration for one tactile key pick-and-sort cycle."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
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
from .tactile_shape_debug import build_debug_panel
from .tactile_detector import DetectionResult, TactileBandDetector


DebugPanelBuilder = Callable[
    [Path],
    tuple[np.ndarray, object | None, dict[str, object]],
]


def _load_tactile_shape_debug_builder() -> DebugPanelBuilder:
    """Return the relocated debug classifier from this control package."""
    return build_debug_panel


@dataclass(frozen=True)
class TactileShapeClassification(KeyClassification):
    """Existing motion classification plus the real debug-script output."""

    shape_label: str
    orientation_deg: float | None
    shape_result: object
    debug_panel: np.ndarray = field(repr=False, compare=False)
    shape_features: dict[str, object] = field(repr=False, compare=False)


@dataclass(frozen=True)
class DebugEdgeEvidence:
    """Line evidence returned by ``tactile_shape_debug.build_debug_panel``."""

    edge_count: int


class TactileShapeDebugClassifierAdapter:
    """Adapt ``tactile_shape_debug`` to the existing classifier interface."""

    def __init__(
        self,
        *,
        minimum_good_edges: int = 0,
        debug_panel_builder: DebugPanelBuilder | None = None,
    ) -> None:
        self._key_classifier = EmbossedFeatureClassifier(
            minimum_good_edges=minimum_good_edges
        )
        self._build_debug_panel = (
            debug_panel_builder or _load_tactile_shape_debug_builder()
        )

    def classify(
        self,
        image: np.ndarray | DetectionResult,
    ) -> TactileShapeClassification:
        """Run the real debug classifier and adapt its output to a task label."""
        frame = (
            image
            if isinstance(image, np.ndarray)
            else image.source_image
        )

        # build_debug_panel() deliberately accepts a Path. A lossless temporary
        # PNG adapts the existing in-memory camera frame without changing that
        # script's interface or classification logic.
        with tempfile.TemporaryDirectory(
            prefix="tactile_shape_classifier_"
        ) as temporary_dir:
            frame_path = Path(temporary_dir) / "captured_frame.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(
                    "Could not create the temporary classifier input image"
                )
            debug_panel, shape_result, features = self._build_debug_panel(
                frame_path
            )

        if shape_result is None:
            raise RuntimeError(
                "tactile_shape_debug could not classify the captured frame"
            )

        try:
            debug_edge_count = int(features["num_edges"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "tactile_shape_debug did not return a valid num_edges value"
            ) from exc
        if debug_edge_count < 0:
            raise RuntimeError(
                "tactile_shape_debug returned a negative num_edges value"
            )

        key_classification = self._key_classifier.classify(
            DebugEdgeEvidence(edge_count=debug_edge_count)
        )
        return TactileShapeClassification(
            label=key_classification.label,
            edge_count=key_classification.edge_count,
            minimum_good_edges=key_classification.minimum_good_edges,
            shape_label=shape_result.label,
            orientation_deg=shape_result.orientation_deg,
            shape_result=shape_result,
            debug_panel=debug_panel,
            shape_features=dict(features),
        )


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

    def destination_for(self, label: KeyLabel) -> ArmPose:
        if label is KeyLabel.GOOD:
            return self.good_drop
        return self.defect_drop

    def destination_for_classifier_output(
        self,
        classification_output: KeyClassification | None,
    ) -> ArmPose:
        """Use the adapter's GOOD/DEFECT output as the motion-routing key."""
        if classification_output is None:
            raise ValueError(
                "A successful tactile-shape classification is required for motion"
            )
        return self.destination_for(classification_output.label)

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
    classification: TactileShapeClassification
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
        classifier: TactileShapeDebugClassifierAdapter,
        poses: SortingPoses,
        *,
        output_dir: Path | str = "sorting_results",
        gripper_max_current: int | None = 60,
        tactile_settle_delay_sec: float = 0.5,
        return_home: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        if gripper_max_current is not None and gripper_max_current <= 0:
            raise ValueError("gripper_max_current must be positive")
        if tactile_settle_delay_sec < 0.0:
            raise ValueError("tactile_settle_delay_sec cannot be negative")

        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.classifier = classifier
        self.poses = poses
        self.output_dir = Path(output_dir)
        self.gripper_max_current = gripper_max_current
        self.tactile_settle_delay_sec = tactile_settle_delay_sec
        self.return_home = return_home
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

            self._log("Classifying tactile shape and line evidence")
            classification = self.classifier.classify(frame)
            self._log(
                f"Classification: {classification.label.value} "
                f"({classification.edge_count} debug-script edges); "
                f"shape={classification.shape_label}; "
                f"orientation={classification.orientation_deg}"
            )

            # Retain the existing detector only for its legacy annotated and
            # preprocessed inspection artifacts. Its edge_count is not used to
            # choose GOOD/DEFECT or the motion destination.
            self._log("Generating legacy edge-detection artifacts")
            detection = self.detector.detect(frame)

            paths = self._save_inspection_artifacts(
                frame,
                detection,
                classification,
            )

            # self._log("Lifting key clear of stand")
            # self.robot.arm.move_to_pose(self.poses.pick_lift)

            drop_pose = self.poses.destination_for_classifier_output(
                classification
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
        classification: TactileShapeClassification,
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
                    f"shape_label={classification.shape_label}",
                    f"orientation_deg={classification.orientation_deg}",
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
