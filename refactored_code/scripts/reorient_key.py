#!/usr/bin/env python3
"""Inspect one key with DIGIT, discard defects, and place good keys upright.

Before hardware execution:
1. Put ``reorientation_task.py`` in ``control/``.
2. Export its public classes from ``control/__init__.py``.
3. Replace every sample pose below with measured poses.
4. Replace the sample gripper ticks.
5. Calibrate the DIGIT-to-wrist angle mapping.
6. Set all three calibration flags to ``True``.
7. Run ``python -m scripts.reorient_key --validate-poses``.
8. Run the real cycle with ``--port`` and ``--digit-serial``.
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
    KeyReorientationTask,
    OrientationCalibration,
    ReorientationMotionConfig,
    ReorientationPoses,
    RobotHardware,
    TactileBandDetector,
)
from sensors import DigitCamera, DigitCameraConfig


# ---------------------------------------------------------------------------
# HARD-CODED CELL CALIBRATION
# ---------------------------------------------------------------------------
# These are EXAMPLE values only. Do not run the physical arm until every pose,
# gripper limit, and orientation setting has been measured on your setup.
POSITIONS_CALIBRATED = False
GRIPPER_CALIBRATED = False
ORIENTATION_CALIBRATED = False


REORIENTATION_POSES = ReorientationPoses(
    # Safe resting pose.
    home=ArmPose.from_cm_degrees(
        x_cm=10.0,
        y_cm=0.0,
        z_cm=9.0,
        pitch_deg=0.0,
    ),

    # The user places the key between the fingers at this pose.
    grasp=ArmPose.from_cm_degrees(
        x_cm=19.5,
        y_cm=0.0,
        z_cm=15.0,
        pitch_deg=0.0,
    ),

    # Raise the key high enough to clear the table and nearby hardware.
    lift=ArmPose.from_cm_degrees(
        x_cm=19.5,
        y_cm=0.0,
        z_cm=15.0,
        pitch_deg=0.0,
    ),

    # Defect path: move to one side first, then sweep towards the left.
    # Confirm the sign of the y-axis on your physical setup.
    defect_windup=ArmPose.from_cm_degrees(
        x_cm=18.0,
        y_cm=-8.0,
        z_cm=10.0,
        pitch_deg=0.0,
    ),
    defect_release=ArmPose.from_cm_degrees(
        x_cm=19.5,
        y_cm=-15.0,
        z_cm=10.0,
        pitch_deg=0.0,
    ),

    # Good-key path: same x/y and pitch for both poses; only z should normally
    # change so the key is lowered vertically onto its flat base.
    good_above=ArmPose.from_cm_degrees(
        x_cm=19.5,
        y_cm=15.0,
        z_cm=12.0,
        pitch_deg=0.0,
    ),
    good_place=ArmPose.from_cm_degrees(
        x_cm=18.0,
        y_cm=15.0,
        z_cm=10.0,
        pitch_deg=0.0,
    ),
)


GRIPPER_CALIBRATION = GripperCalibration(
    closed_ticks=2250,  # TODO: replace with measured grasp/contact position
    open_ticks=2000,    # TODO: replace with measured fully-open position
)


ORIENTATION_CALIBRATION = OrientationCalibration(
    # Angle reported by the detector when the embossed line is visually
    # vertical in the saved OpenCV image. Often 90 degrees, but verify it.
    image_vertical_deg=90.0,

    # Use +1 if a positive image correction needs a positive joint-4 command.
    # Change to -1 if the wrist rotates in the opposite physical direction.
    wrist_direction=1,

    # Small constant correction for residual alignment error after testing.
    wrist_zero_offset_deg=0.0,

    # Safety limit for one automatically calculated wrist correction.
    max_abs_wrist_command_deg=90.0,
)


MOTION_CONFIG = ReorientationMotionConfig(
    wrist_speed=30,
    wrist_step_deg=1.0,
    wrist_step_delay_sec=0.02,

    placement_speed=20,
    placement_step_deg=1.0,
    placement_step_delay_sec=0.04,

    throw_speed=100,
    throw_step_deg=4.0,
    throw_step_delay_sec=0.01,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-poses",
        action="store_true",
        help="Check inverse kinematics for all fixed poses without connecting hardware",
    )
    parser.add_argument(
        "--port",
        help="Dynamixel serial port, for example /dev/ttyUSB0",
    )
    parser.add_argument(
        "--digit-serial",
        help="Actual DIGIT serial number reported by the SDK",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reorientation_results"),
        help="Directory for tactile images and result metadata",
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
        help="Optional DIGIT stream name, for example QVGA",
    )
    parser.add_argument(
        "--digit-fps",
        help="Optional FPS key for the selected stream, for example 30fps",
    )
    return parser.parse_args()


def validate_poses_without_hardware() -> None:
    """Check that all fixed poses have inverse-kinematics solutions."""
    kinematics = ArmKinematics()

    for name, pose in REORIENTATION_POSES.named().items():
        joints = kinematics.inverse(pose)
        joint_degrees = [round(math.degrees(q), 2) for q in joints]
        print(f"{name:18s}: reachable; joints_deg={joint_degrees}")

    print(
        "\nFixed poses are reachable. The angle-adjusted good-key poses are "
        "validated again during each run after the DIGIT angle is measured."
    )


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
    if not ORIENTATION_CALIBRATED:
        missing.append("ORIENTATION_CALIBRATED")

    if missing:
        raise SystemExit(
            "Hardware execution is blocked. Replace all sample calibration "
            "values, then set these flags to True: "
            + ", ".join(missing)
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

    # Imported only for a real run so pose validation can work on a computer
    # without the hardware driver's dependencies.
    from interface.dynamixel_driver import DynamixelDriver

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
        task = KeyReorientationTask(
            robot=robot,
            camera=camera,
            detector=detector,
            classifier=classifier,
            poses=REORIENTATION_POSES,
            orientation=ORIENTATION_CALIBRATION,
            motion=MOTION_CONFIG,
            output_dir=args.output_dir,
            gripper_max_current=args.max_gripper_current,
            blank_image=blank_image,
        )
        result = task.run_once()

    print(f"Completed: {result.classification.label.value}")
    print(f"Detected edges: {result.classification.edge_count}")

    if result.wrist_command_deg is not None:
        print(f"Measured line angle: {result.measured_line_angle_deg:.2f} deg")
        print(f"Image correction: {result.image_correction_deg:+.2f} deg")
        print(f"Wrist command: {result.wrist_command_deg:+.2f} deg")

    print(f"Raw image: {result.raw_image_path}")
    print(f"Annotated image: {result.annotated_image_path}")
    print(f"Preprocessed image: {result.preprocessed_image_path}")


if __name__ == "__main__":
    main()
