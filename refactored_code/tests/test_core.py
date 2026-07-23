from __future__ import annotations

import math

import numpy as np
import pytest

from control import (
    ArmController,
    ArmKinematics,
    GripperCalibration,
    GripperController,
    MotionConfig,
    TactileBandDetector,
    Workspace,
)
from control.kinematics import interpolate_joint_steps


class FakeDriver:
    def __init__(self) -> None:
        self.default_speed = 60
        self.joints = [0.0, 0.0, 0.0, 0.0]
        self.speeds = [60, 60, 60, 60]
        self.gripper_position = 3000
        self.gripper_current_ma = 25.0
        self.connected = False

    def connect(self, port: str) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def close_port_keep_torque(self) -> None:
        self.connected = False

    def read_joint_angles(self):
        return list(self.joints)

    def set_joint_angles(self, angles):
        self.joints = list(angles[:4])

    def set_joint_speeds(self, speeds):
        self.speeds = list(speeds[:4])

    def read_gripper_position(self):
        return self.gripper_position

    def set_gripper_position(self, position, max_current=None):
        self.gripper_position = int(position)

    def set_gripper_current(self, current):
        self.gripper_current_ma = float(current)

    def read_gripper_current_ma(self):
        return self.gripper_current_ma


def test_kinematics_forward_inverse_round_trip() -> None:
    kinematics = ArmKinematics()
    joints = (0.1, 0.5, -0.8, 0.3)
    pose = kinematics.forward(joints)
    solved = kinematics.inverse(pose)
    reconstructed = kinematics.forward(solved)
    assert math.dist(
        (pose.x, pose.y, pose.z),
        (reconstructed.x, reconstructed.y, reconstructed.z),
    ) < 1e-9
    assert reconstructed.angle_rad == pytest.approx(pose.angle_rad)


def test_interpolation_bounds_each_step() -> None:
    waypoints = interpolate_joint_steps(
        (0.0, 0.0, 0.0, 0.0),
        (math.radians(5.0), 0.0, 0.0, 0.0),
        max_step_deg=2.0,
    )
    assert len(waypoints) == 3
    previous = (0.0, 0.0, 0.0, 0.0)
    for waypoint in waypoints:
        assert max(
            abs(math.degrees(end - start))
            for start, end in zip(previous, waypoint)
        ) <= 2.0
        previous = waypoint


def test_arm_controller_uses_injected_driver() -> None:
    driver = FakeDriver()
    kinematics = ArmKinematics()
    target_pose = kinematics.forward((0.0, 0.4, -0.8, 0.4))
    controller = ArmController(
        driver,
        workspace=Workspace(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            min_z=-1.0,
            max_z=1.0,
        ),
        motion=MotionConfig(max_step_deg=5.0, step_delay_sec=0.0),
    )
    result = controller.move_to_pose(target_pose)
    assert result.waypoint_count > 0
    assert tuple(driver.joints) == pytest.approx(result.target_joints)


def test_gripper_controller_respects_calibration() -> None:
    driver = FakeDriver()
    gripper = GripperController(
        driver,
        GripperCalibration(closed_ticks=1800, open_ticks=3200),
        settle_delay_sec=0.0,
    )
    assert gripper.close().position_ticks == 1800
    assert gripper.open().position_ticks == 3200
    with pytest.raises(ValueError):
        gripper.move_to(1000)


def test_blank_tactile_image_returns_result() -> None:
    detector = TactileBandDetector()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert result.edge_count == 0
    assert result.annotated_image.shape == image.shape


def test_embossed_classifier_maps_edge_evidence() -> None:
    from control import EmbossedFeatureClassifier, KeyLabel

    class Evidence:
        def __init__(self, edge_count: int) -> None:
            self.edge_count = edge_count

    classifier = EmbossedFeatureClassifier(minimum_good_edges=2)
    assert classifier.classify(Evidence(0)).label is KeyLabel.DEFECT
    assert classifier.classify(Evidence(2)).label is KeyLabel.GOOD
    assert classifier.classify(Evidence(4)).label is KeyLabel.GOOD


def test_digit_camera_uses_injected_device_and_warms_up() -> None:
    from sensors import DigitCamera, DigitCameraConfig

    class FakeDigit:
        STREAMS = {
            "QVGA": {
                "fps": {"30fps": 30},
            }
        }

        def __init__(self, serial: str) -> None:
            self.serial = serial
            self.connected = False
            self.frames_read = 0
            self.resolution = None
            self.fps = None

        def connect(self) -> None:
            self.connected = True

        def disconnect(self) -> None:
            self.connected = False

        def get_frame(self, transpose: bool = False):
            self.frames_read += 1
            self.last_transpose = transpose
            return np.zeros((24, 32, 3), dtype=np.uint8)

        def set_resolution(self, resolution) -> None:
            self.resolution = resolution

        def set_fps(self, fps) -> None:
            self.fps = fps

        def set_intensity(self, intensity: int) -> int:
            self.intensity = intensity
            return intensity

    made = []

    def factory(serial: str):
        device = FakeDigit(serial)
        made.append(device)
        return device

    camera = DigitCamera(
        DigitCameraConfig(
            serial_number="DTEST",
            resolution="QVGA",
            fps="30fps",
            warmup_frames=2,
            stream_settle_delay_sec=0.0,
            flush_frames_per_capture=0,
        ),
        device_factory=factory,
    )
    with camera:
        frame = camera.capture_frame()
        assert frame.shape == (24, 32, 3)
        assert made[0].frames_read == 3
        assert made[0].last_transpose is True
        assert made[0].intensity == 15

        # A second capture on the same connection re-flushes per call, not
        # just once after connect -- otherwise a stale buffered frame from
        # before the gap (e.g. gripper motion time) could be returned.
        frames_before = made[0].frames_read
        camera.capture_frame()
        assert made[0].frames_read == frames_before + 1  # 0 flush + 1 real read
        assert made[0].resolution == FakeDigit.STREAMS["QVGA"]
        assert made[0].fps == 30
    assert not made[0].connected


def test_complete_sorting_task_defect_path(tmp_path) -> None:
    from control import (
        ArmKinematics,
        EmbossedFeatureClassifier,
        GripperCalibration,
        KeyLabel,
        KeySortingTask,
        MotionConfig,
        RobotHardware,
        SortingPoses,
        TactileBandDetector,
        Workspace,
    )

    class FakeCamera:
        def capture_frame(self):
            return np.zeros((120, 160, 3), dtype=np.uint8)

    driver = FakeDriver()
    kinematics = ArmKinematics()
    pose = kinematics.forward((0.0, 0.5, -0.9, 0.4))
    poses = SortingPoses(
        home=pose,
        pick_grasp=pose,
        pick_lift=pose,
        good_drop=pose,
        defect_drop=pose,
    )
    robot = RobotHardware(
        driver,
        port="FAKE",
        gripper_calibration=GripperCalibration(
            closed_ticks=1800,
            open_ticks=3200,
        ),
        kinematics=kinematics,
        workspace=Workspace(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            min_z=-1.0,
            max_z=1.0,
        ),
        motion=MotionConfig(max_step_deg=10.0, step_delay_sec=0.0),
    )
    robot.gripper.settle_delay_sec = 0.0

    with robot:
        task = KeySortingTask(
            robot=robot,
            camera=FakeCamera(),
            detector=TactileBandDetector(),
            classifier=EmbossedFeatureClassifier(minimum_good_edges=2),
            poses=poses,
            output_dir=tmp_path,
            tactile_settle_delay_sec=0.0,
            sleep=lambda _: None,
            log=lambda _: None,
        )
        result = task.run_once()

    assert result.classification.label is KeyLabel.DEFECT
    assert driver.gripper_position == 3200
    assert result.raw_image_path.exists()
    assert result.annotated_image_path.exists()
    assert result.preprocessed_image_path.exists()


def test_wait_for_arrival_proceeds_after_timeout_instead_of_raising() -> None:
    """A stuck arm must not abort the cycle -- it should log a warning and
    let the gripper open anyway, since never releasing is worse than
    releasing slightly early. This is the behaviour that replaced an
    earlier, reverted approach where ArmController.move_to_pose() itself
    raised on timeout for every move, which on real hardware aborted before
    the grasp-pose user-input prompt was ever reached."""
    from control import (
        EmbossedFeatureClassifier,
        KeySortingTask,
        RobotHardware,
        SortingPoses,
    )

    class StuckDriver(FakeDriver):
        def set_joint_angles(self, angles):
            pass  # position never actually updates

    driver = StuckDriver()
    kinematics = ArmKinematics()
    robot = RobotHardware(
        driver,
        port="FAKE",
        gripper_calibration=GripperCalibration(closed_ticks=1800, open_ticks=3200),
        kinematics=kinematics,
        workspace=Workspace(
            min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0, min_z=-1.0, max_z=1.0
        ),
        motion=MotionConfig(max_step_deg=10.0, step_delay_sec=0.0),
    )
    logs: list[str] = []
    pose = kinematics.forward((0.0, 0.5, -0.9, 0.4))
    poses = SortingPoses(home=pose, pick_grasp=pose, pick_lift=pose, good_drop=pose, defect_drop=pose)

    with robot:
        task = KeySortingTask(
            robot=robot,
            camera=None,
            detector=TactileBandDetector(),
            classifier=EmbossedFeatureClassifier(),
            poses=poses,
            release_arrival_timeout_sec=0.05,
            release_arrival_poll_interval_sec=0.01,
            sleep=lambda _: None,
            log=logs.append,
        )
        target_joints = kinematics.inverse(
            kinematics.forward((0.3, 0.2, -0.5, 0.1))
        )
        task._wait_for_arrival(target_joints)  # must not raise

    assert any("releasing anyway" in message for message in logs)
