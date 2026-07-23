"""Control components for tactile key inspection and sorting."""

from .arm_controller import ArmController, ArmMoveResult, MotionConfig, Workspace
from .gripper_controller import GripperCalibration, GripperController, GripperState
from .key_classifier import (
    EmbossedFeatureClassifier,
    KeyClassification,
    KeyLabel,
)
from .kinematics import ArmGeometry, ArmKinematics, ArmPose
from .robot_hardware import RobotHardware
from .sorting_task import KeySortingTask, SortRunResult, SortingPoses
from .tactile_detector import (
    DetectionRecord,
    DetectionResult,
    DetectorConfig,
    TactileBandDetector,
)
from .reorientation_task import (
    KeyReorientationTask,
    OrientationCalibration,
    ReorientationMotionConfig,
    ReorientationPoses,
    ReorientationRunResult,
)

__all__ = [
    "ArmController",
    "ArmGeometry",
    "ArmKinematics",
    "ArmMoveResult",
    "ArmPose",
    "DetectionRecord",
    "DetectionResult",
    "EmbossedFeatureClassifier",
    "DetectorConfig",
    "GripperCalibration",
    "GripperController",
    "GripperState",
    "KeyClassification",
    "KeyLabel",
    "KeySortingTask",
    "MotionConfig",
    "RobotHardware",
    "SortRunResult",
    "SortingPoses",
    "TactileBandDetector",
    "Workspace",
    "KeyReorientationTask",
    "OrientationCalibration",
    "ReorientationMotionConfig",
    "ReorientationPoses",
    "ReorientationRunResult",
]
