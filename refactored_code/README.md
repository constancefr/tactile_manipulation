# Refactored tactile sorting components

This folder reorganises the supplied arm, gripper, kinematics, and DIGIT edge-detection code into reusable components. It intentionally does **not** implement the final pick/classify/sort sequence yet.

## Structure

- `control/kinematics.py` — pure kinematics and joint interpolation.
- `control/arm_controller.py` — callable Cartesian arm motion methods.
- `control/gripper_controller.py` — calibrated, callable gripper methods.
- `control/robot_hardware.py` — one shared Dynamixel connection for arm and gripper.
- `control/tactile_detector.py` — in-memory and file-based tactile edge detection.
- `scripts/` — thin command-line wrappers for standalone hardware/vision tests.
- `tests/` — hardware-free tests using a fake driver.

The existing project must still provide:

```text
interface/dynamixel_driver.py
interface/ports.py
```

Run commands from the project root with module syntax, for example:

```bash
python -m scripts.move_to_pose --x-cm 20 --y-cm 12.5 --z-cm 10 --pitch-deg 0 --port /dev/ttyUSB0
python -m scripts.gripper_control --port /dev/ttyUSB0 --close --max-current 60
python -m scripts.detect_tactile_bands path/to/image.png --save-debug
```


## Callables for the future task script

```python
from control import (
    DetectorConfig,
    GripperCalibration,
    RobotHardware,
    TactileBandDetector,
)
from interface.dynamixel_driver import DynamixelDriver

calibration = GripperCalibration(
    closed_ticks=1800,  # replace with measured values
    open_ticks=3200,
)

with RobotHardware(
    DynamixelDriver(default_speed=60),
    port="/dev/ttyUSB0",
    gripper_calibration=calibration,
) as robot:
    robot.arm.move_to_cm(20.0, 12.5, 10.0, pitch_deg=0.0)
    robot.gripper.close(max_current=60)

    # `frame` should be the BGR NumPy image returned by the DIGIT capture code.
    detector = TactileBandDetector(DetectorConfig())
    result = detector.detect(frame)
    print(result.edge_count)
```

The detector returns raw evidence (`edge_count`, edges, annotated image, debug images). It does not assign `"good"` or `"defect"`; that mapping should be a separate classifier once the rule is reliable.

## Important integration notes

1. Arm and gripper controllers share one connected driver. The final workflow should not repeatedly open the same serial port.
2. Replace the placeholder gripper limits before hardware use.
3. Confirm the workspace limits and hard-coded poses on the physical setup.
4. DIGIT camera acquisition code was not present in the supplied files. The detector accepts a NumPy BGR frame so capture can be added independently.
5. The earlier detector treated two or four edges as expected cases. A flat-versus-one-embossed-line task is instead likely to produce zero versus two edges, so label semantics are deliberately not baked into the refactor.

## Complete pick–inspect–sort task

The integration added for the complete task is split into three reusable parts:

- `sensors/digit_camera.py` — connects to one DIGIT by serial number and returns an in-memory BGR frame.
- `control/key_classifier.py` — temporary edge-count mapping from detector evidence to `good`/`defect`.
- `control/sorting_task.py` — the hardware-independent sequence controller.
- `scripts/sort_key.py` — the executable configuration and entry point.

The sequence is:

1. Open the gripper.
2. Move above the key.
3. Descend to the fixed grasp pose.
4. Close the gripper using the configured current limit.
5. Capture one DIGIT frame and analyse it while the key remains on the stand.
6. Classify the key as `good` or `defect`.
7. Lift the key clear of the stand.
8. Move above the corresponding bucket.
9. Lower, release, retreat, and return home.

### Required calibration

`scripts/sort_key.py` contains sample coordinates and sample gripper ticks. Hardware execution is deliberately blocked until you replace them and set:

```python
POSITIONS_CALIBRATED = True
GRIPPER_CALIBRATED = True
```

Validate inverse kinematics before connecting hardware:

```bash
python -m scripts.sort_key --validate-poses
```

Then perform one real cycle:

```bash
python -m scripts.sort_key \
  --port /dev/ttyUSB0 \
  --digit-serial D12345 \
  --output-dir sorting_results
```
OR:
```
python -m scripts.sort_key \
    --port /dev/ttyUSB0 \
    --digit-serial D20908 \
    --minimum-good-edges 2 \
    --max-gripper-current 60 \
    --output-dir sorting_results
```

The script saves the raw, annotated, and preprocessed DIGIT images for every cycle. A sensing or classification error never chooses a bucket; if such an error occurs after grasping, the gripper remains closed for operator recovery.

### Temporary label rule

The current `EmbossedFeatureClassifier` uses this provisional mapping:

```text
edge_count >= 2  -> good
edge_count < 2   -> defect
```

This is isolated from the robot workflow so it can be replaced by the final classifier without changing the motion sequence.
