#!/usr/bin/env python3
"""Pick one key, inspect it with DIGIT, and sort it into a fixed bucket.

Before hardware execution:
1. Replace every sample pose in ``SORTING_POSES`` with measured cell poses.
2. Replace the sample gripper ticks in ``GRIPPER_CALIBRATION``.
3. Set both calibration flags to ``True``.
4. Run ``python -m scripts.sort_key --validate-poses``.
5. Run the real cycle with ``--port`` and ``--digit-serial``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2

from control import (
    ArmKinematics,
    ArmPose,
    EmbossedFeatureClassifier,
    GripperCalibration,
    KeySortingTask,
    RobotHardware,
    SortingPoses,
    TactileBandDetector,
)
from sensors import DigitCamera, DigitCameraConfig


# ---------------------------------------------------------------------------
# HARD-CODED CELL CALIBRATION
# ---------------------------------------------------------------------------
# These coordinates are safe-looking examples only. They are not measurements
# of your setup. Keep execution blocked until every value has been calibrated.
POSITIONS_CALIBRATED = True
GRIPPER_CALIBRATED = False

SORTING_POSES = SortingPoses(
    home=ArmPose.from_cm_degrees(10, 0.0, 9, 0.0),
    # pick_approach=ArmPose.from_cm_degrees(20.0, 0.0, 14.0, 0.0),
    pick_grasp=ArmPose.from_cm_degrees(10, 0.0, 9, 0.0),
    # pick_grasp=ArmPose.from_cm_degrees(20.0, 0.0, 9.0, 0.0),
    # pick_lift=ArmPose.from_cm_degrees(20.0, 0.0, 18.0, 0.0),
    # Coordinate signs depend on how your arm's frame is mounted. Verify which
    # side is physically right/left before setting POSITIONS_CALIBRATED=True.
    # good_approach=ArmPose.from_cm_degrees(18.0, -12.0, 18.0, 0.0),
    good_drop=ArmPose.from_cm_degrees(19.5, 15.0, 5.0, 0.0),
    # defect_approach=ArmPose.from_cm_degrees(18.0, 12.0, 18.0, 0.0),
    defect_drop=ArmPose.from_cm_degrees(19.5, -15.0, 5.0, 0.0),
)

GRIPPER_CALIBRATION = GripperCalibration(
    closed_ticks=2250,
    open_ticks=2000,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-poses",
        action="store_true",
        help="Check inverse kinematics for all hardcoded poses without hardware",
    )
    parser.add_argument("--port", help="Dynamixel serial port, e.g. /dev/ttyUSB0")
    parser.add_argument(
        "--digit-serial",
        help="DIGIT serial number printed on the sensor, e.g. D12345",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sorting_results"),
        help="Directory for raw, annotated, and preprocessed tactile images",
    )
    parser.add_argument(
        "--blank-image",
        type=Path,
        help="Optional no-contact DIGIT reference image",
    )
    parser.add_argument(
        "--minimum-good-edges",
        type=int,
        default=2,
        help="Temporary rule: classify as good at or above this edge count",
    )
    parser.add_argument(
        "--max-gripper-current",
        type=int,
        default=60,
        help="Current cap used for gripper position commands",
    )
    parser.add_argument(
        "--digit-resolution",
        help="Optional DIGIT stream name, e.g. QVGA; defaults to SDK setting",
    )
    parser.add_argument(
        "--digit-fps",
        help="Optional FPS key for the selected stream, e.g. 30fps",
    )
    return parser.parse_args()


def validate_poses_without_hardware() -> None:
    kinematics = ArmKinematics()
    for name, pose in SORTING_POSES.named().items():
        joints = kinematics.inverse(pose)
        joint_degrees = [round(math.degrees(q), 2) for q in joints]
        print(f"{name:16s}: reachable; joints_deg={joint_degrees}")


def load_blank_image(path: Path | None):
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read blank DIGIT image: {path}")
    return image


def require_hardware_calibration() -> None:
    missing = []
    if not POSITIONS_CALIBRATED:
        missing.append("POSITIONS_CALIBRATED")
    if not GRIPPER_CALIBRATED:
        missing.append("GRIPPER_CALIBRATED")
    if missing:
        raise SystemExit(
            "Hardware execution is blocked. Replace the sample coordinates/ticks "
            f"and set these flags to True: {', '.join(missing)}"
        )


def main() -> None:
    args = parse_args()

    if args.validate_poses:
        validate_poses_without_hardware()
        return

    require_hardware_calibration()
    if not args.port:
        raise SystemExit("--port is required for hardware execution")
    if not args.digit_serial:
        raise SystemExit("--digit-serial is required for hardware execution")

    blank_image = load_blank_image(args.blank_image)

    # Import only for a real hardware run so pose validation still works on a
    # computer that does not have the arm driver's dependencies installed.
    from ..interface.dynamixel_driver import DynamixelDriver

    detector = TactileBandDetector()
    classifier = EmbossedFeatureClassifier(
        minimum_good_edges=args.minimum_good_edges
    )
    digit_config = DigitCameraConfig(
        serial_number=args.digit_serial,
        resolution=args.digit_resolution,
        fps=args.digit_fps,
    )

    robot = RobotHardware(
        DynamixelDriver(default_speed=60),
        port=args.port,
        gripper_calibration=GRIPPER_CALIBRATION,
        keep_torque_on_close=True,
    )

    with robot, DigitCamera(digit_config) as camera:
        task = KeySortingTask(
            robot=robot,
            camera=camera,
            detector=detector,
            classifier=classifier,
            poses=SORTING_POSES,
            output_dir=args.output_dir,
            gripper_max_current=args.max_gripper_current,
            blank_image=blank_image,
        )
        result = task.run_once()

    print(f"Completed: {result.classification.label.value}")
    print(f"Detected edges: {result.classification.edge_count}")
    print(f"Raw image: {result.raw_image_path}")
    print(f"Annotated image: {result.annotated_image_path}")


if __name__ == "__main__":
    main()
