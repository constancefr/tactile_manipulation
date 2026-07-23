"""High-level orchestration for one tactile key pick-and-sort cycle."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from .key_classifier import (
    EmbossedFeatureClassifier,
    KeyClassification,
    KeyLabel,
    draw_classification_overlay,
)
from .kinematics import ArmPose
from .robot_hardware import RobotHardware
from .tactile_detector import DetectionResult, TactileBandDetector


class TactileCameraProtocol(Protocol):
    def capture_frame(self) -> np.ndarray: ...


def release_pitch_rad(base_pitch_rad: float, image_angle_deg: float) -> float:
    """Best-effort pitch compensation so the grasped rectangle's edge is
    vertical after release.

    The DIGIT sensor is mounted inline with the gripper so that the tactile
    image's long axis (0 deg in this project's angle convention, see
    ``tactile_detector.Segment.angle_rad``) runs along the arm's pitch
    rotation axis, and the short axis (90 deg) runs along the
    pitch-controlled forward direction.

    This arm has only 4 DOF: base yaw plus one pitch confined to the arm's
    vertical plane -- there is no independent wrist-roll axis. An edge at
    image angle 90 deg has no component along the (fixed) pitch axis, so
    rotating pitch alone can align it exactly vertical. For any other image
    angle, the edge has a fixed component along the pitch axis itself, which
    no choice of pitch can rotate away -- exact verticality is only reachable
    when image_angle_deg == 90. This still applies the same linear
    correction as the best available single-DOF approximation for other
    angles. Verify against the real arm and revisit if a wrist-roll axis is
    ever added.
    """
    return base_pitch_rad + math.radians(90.0 - image_angle_deg)


@dataclass(frozen=True)
class SortingPoses:
    """All fixed poses required by the sorting cell."""

    home: ArmPose
    # pick_approach: ArmPose
    pick_grasp: ArmPose
    pick_lift: ArmPose
    # good_approach: ArmPose
    good_drop: ArmPose
    # defect_approach: ArmPose
    defect_drop: ArmPose

    def destination_for(self, label: KeyLabel) -> ArmPose:
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
            # "pick_approach": self.pick_approach,
            "pick_grasp": self.pick_grasp,
            "pick_lift": self.pick_lift,
            # "good_approach": self.good_approach,
            "good_drop": self.good_drop,
            # "defect_approach": self.defect_approach,
            "defect_drop": self.defect_drop,
        }


@dataclass(frozen=True)
class SortRunResult:
    classification: KeyClassification
    detection: DetectionResult
    angle_deg: float | None
    release_pose: ArmPose
    raw_image_path: Path
    reference_image_path: Path
    annotated_image_path: Path
    preprocessed_image_path: Path
    debug_dir: Path | None


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
        save_debug: bool = False,
        release_arrival_tolerance_deg: float = 2.0,
        release_arrival_timeout_sec: float = 5.0,
        release_arrival_poll_interval_sec: float = 0.1,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
        input_fn: Callable[[str], str] = input,
    ) -> None:
        if gripper_max_current is not None and gripper_max_current <= 0:
            raise ValueError("gripper_max_current must be positive")
        if tactile_settle_delay_sec < 0.0:
            raise ValueError("tactile_settle_delay_sec cannot be negative")
        if release_arrival_tolerance_deg <= 0.0:
            raise ValueError("release_arrival_tolerance_deg must be positive")
        if release_arrival_timeout_sec <= 0.0:
            raise ValueError("release_arrival_timeout_sec must be positive")
        if release_arrival_poll_interval_sec <= 0.0:
            raise ValueError("release_arrival_poll_interval_sec must be positive")

        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.classifier = classifier
        self.poses = poses
        self.output_dir = Path(output_dir)
        self.gripper_max_current = gripper_max_current
        self.tactile_settle_delay_sec = tactile_settle_delay_sec
        self.return_home = return_home
        self.save_debug = save_debug
        self.release_arrival_tolerance_deg = release_arrival_tolerance_deg
        self.release_arrival_timeout_sec = release_arrival_timeout_sec
        self.release_arrival_poll_interval_sec = release_arrival_poll_interval_sec
        self._sleep = sleep
        self._log = log
        self._input = input_fn

    def validate_poses(self) -> None:
        """Run inverse kinematics for every fixed pose before moving hardware."""
        for name, pose in self.poses.named().items():
            self.robot.arm.solve_pose(pose)
            self._log(f"Pose validated: {name}")

    def _wait_for_arrival(self, target_joints: tuple[float, float, float, float]) -> None:
        """Best-effort wait for the arm to physically reach the release pose
        before the gripper opens.

        Bounded and non-fatal by design: it polls actual joint position
        (move_to_pose() only guarantees the last command was *sent*, not
        that the physical arm has arrived) but proceeds anyway, with a
        logged warning, if it doesn't converge within the timeout, rather
        than raising. An earlier version of this used a hard timeout inside
        ArmController.move_to_pose() itself, applied to every move including
        the initial move to the grasp pose -- on real hardware that timed
        out before the user-input prompt was ever reached, aborting the
        whole cycle. Scoping this to only the release step, and failing
        open instead of raising, avoids that: the worst case here is
        releasing slightly before fully settled, not losing the object by
        never releasing at all.
        """
        tolerance = math.radians(self.release_arrival_tolerance_deg)
        deadline = time.monotonic() + self.release_arrival_timeout_sec
        while True:
            current_joints = self.robot.arm.read_joint_angles()
            errors = [
                abs(math.atan2(math.sin(actual - target), math.cos(actual - target)))
                for actual, target in zip(current_joints, target_joints)
            ]
            if all(error <= tolerance for error in errors):
                return
            if time.monotonic() >= deadline:
                self._log(
                    "Warning: arm did not confirm arrival at the release "
                    f"pose within {self.release_arrival_timeout_sec}s "
                    f"(joint errors, deg: {[round(math.degrees(e), 2) for e in errors]}, "
                    f"tolerance: {self.release_arrival_tolerance_deg} deg) -- "
                    "releasing anyway"
                )
                return
            self._sleep(self.release_arrival_poll_interval_sec)

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
            # self._log("Moving above key")
            # self.robot.arm.move_to_pose(self.poses.pick_approach)

            self._log("Descending to grasp pose")
            self.robot.arm.move_to_pose(self.poses.pick_grasp)

            self._log("Opening gripper")
            self.robot.gripper.open(max_current=self.gripper_max_current)

            # Wait for user to place the key in the gripper's grasping area
            try:
                self._input("Press Enter when ready to grasp the key...")
            except (EOFError, OSError):
                # Non-interactive environments such as pytest capture stdin.
                # Skip the pause there so the task remains testable.
                pass

            # Reference frame is captured live, right before the gripper
            # closes, rather than loaded from a pre-saved file: lighting and
            # exposure drift session to session, so a stale static reference
            # produces an unreliable diff (see live_tactile_detect.py).
            self._log("Capturing DIGIT reference frame (no object gripped)")
            reference_frame = self.camera.capture_frame()

            self._log("Closing gripper around key")
            self.robot.gripper.close(max_current=self.gripper_max_current)
            holding_key = True

            if self.tactile_settle_delay_sec:
                self._sleep(self.tactile_settle_delay_sec)

            self._log("Capturing DIGIT tactile image (object gripped)")
            frame = self.camera.capture_frame()

            self._log("Detecting embossed-feature edges")
            detection = self.detector.detect(
                frame,
                blank_image=reference_frame,
            )
            classification = self.classifier.classify(detection)
            angle_deg = (
                math.degrees(detection.dominant_angle_rad)
                if detection.dominant_angle_rad is not None
                else None
            )
            self._log(
                f"Classification: {classification.label.value} "
                f"({classification.edge_count} detected edges, "
                f"angle={angle_deg:.1f} deg)"
                if angle_deg is not None
                else f"Classification: {classification.label.value} "
                f"({classification.edge_count} detected edges, angle=n/a)"
            )

            paths = self._save_inspection_artifacts(
                reference_frame,
                frame,
                detection,
                classification,
                angle_deg,
            )

            self._log("Lifting key clear of stand")
            self.robot.arm.move_to_pose(self.poses.pick_lift)

            drop_pose = self.poses.destination_for(
                classification.label
            )
            self._log(
                f"Moving to {classification.label.value} bucket approach"
            )
            # self.robot.arm.move_to_pose(approach_pose)

            # Only a detected rectangle gives a trustworthy edge angle
            # (classifier.label is GOOD iff edge_count >= minimum_good_edges,
            # which also guarantees dominant_angle_rad came from real
            # selected edges, not just background noise). Otherwise leave
            # the bucket pose's own configured pitch untouched.
            release_pose = drop_pose
            if classification.label is KeyLabel.GOOD and angle_deg is not None:
                adjusted_pitch = release_pitch_rad(drop_pose.angle_rad, angle_deg)
                candidate_pose = replace(drop_pose, angle_rad=adjusted_pitch)
                try:
                    self.robot.arm.solve_pose(candidate_pose)
                    release_pose = candidate_pose
                    self._log(
                        "Adjusted release pitch to "
                        f"{math.degrees(adjusted_pitch):.1f} deg so the "
                        f"rectangle (image angle {angle_deg:.1f} deg) is "
                        "vertical on release"
                    )
                except ValueError as exc:
                    self._log(
                        "Pitch-adjusted release pose is unreachable "
                        f"({exc}); releasing at the bucket's default pitch "
                        "instead"
                    )

            self._log("Lowering key")
            move_result = self.robot.arm.move_to_pose(release_pose)
            self._wait_for_arrival(move_result.target_joints)

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
                angle_deg=angle_deg,
                release_pose=release_pose,
                raw_image_path=paths[0],
                reference_image_path=paths[1],
                annotated_image_path=paths[2],
                preprocessed_image_path=paths[3],
                debug_dir=paths[4],
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
        reference_frame: np.ndarray,
        frame: np.ndarray,
        detection: DetectionResult,
        classification: KeyClassification,
        angle_deg: float | None,
    ) -> tuple[Path, Path, Path, Path, Path | None]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.output_dir / f"{timestamp}_{classification.label.value}"
        run_dir.mkdir(parents=True, exist_ok=False)

        raw_path = run_dir / "digit_raw.png"
        reference_path = run_dir / "digit_reference.png"
        annotated_path = run_dir / "digit_annotated.png"
        preprocessed_path = run_dir / "digit_preprocessed.png"

        self._write_image(raw_path, frame)
        self._write_image(reference_path, reference_frame)
        self._write_image(
            annotated_path,
            draw_classification_overlay(
                detection.annotated_image, classification, angle_deg
            ),
        )
        self._write_image(preprocessed_path, detection.enhanced_image)

        debug_dir = None
        if self.save_debug:
            debug_dir = run_dir / "debug"
            self.detector.save_debug_images(detection, debug_dir)

        metadata = run_dir / "result.txt"
        metadata.write_text(
            "\n".join(
                [
                    f"label={classification.label.value}",
                    f"edge_count={classification.edge_count}",
                    f"minimum_good_edges={classification.minimum_good_edges}",
                    f"angle_deg={angle_deg if angle_deg is not None else 'n/a'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return raw_path, reference_path, annotated_path, preprocessed_path, debug_dir

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write image: {path}")
