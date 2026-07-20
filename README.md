# Python

A ROS2 (`rclpy`) package that wraps the OpenManipulator-X in a clean, reusable interface: kinematics math, a Dynamixel serial driver, and ROS2 nodes for both the real hardware and a simulated backend. 

## Where to start

- **`kinematics.py`** — pure math, no ROS or hardware dependency. `ArmGeometry` holds the arm's link lengths/joint offsets, `ArmKinematics.forward`/`.inverse` convert between joint angles and end-effector pose (`ArmPose`: x, y, z, pitch angle). Also has the radian ↔ Dynamixel-tick conversions (`rad_to_dxl`/`dxl_to_rad`) and `synchronized_joint_speeds` (scales per-joint speeds so all 4 joints arrive at their target at the same time). Read this first — it's the part worth understanding regardless of which backend you use.
- **`sim_arm_node.py`** — a ROS2 node (`sim_arm_node`) backed by an in-memory `SimBackend` (no serial port, no real motors). Run this to develop and test your ROS-side logic (perception → decision → arm command) before touching hardware.
- **`hardware_arm_node.py`** — the same node interface (`hardware_arm_node`), backed by `DynamixelDriver` talking to the real arm over serial. Swap to this once your logic works against the sim node.
- **`arm_node_base.py`** — shared logic both nodes above build on: subscribes to `/arm/command` (move / gripper / stop), publishes `/arm/state`, and exposes `/arm/list_ports` and `/arm/set_connection` services. Defines the `ArmBackend` protocol that `SimBackend` and `DynamixelDriver` both implement — this is the seam to use if you ever need a third backend.
- **`dynamixel_driver.py`** — low-level serial driver over `dynamixel_sdk`: connects/initializes the 5 motors (4 joints + gripper), reads/writes joint angles and speeds, handles retries and a lock around the serial bus so concurrent reads/writes don't corrupt each other.
- **`ports.py`** — lists likely serial devices (`/dev/ttyUSB*`, `/dev/ttyACM*`) and guesses which one is the arm.

## Notes / prerequisites

- Needs a ROS2 environment with `rclpy` and a `tactile_interfaces` package providing the `ArmCommand`/`ArmState`/`ArmPose` messages and `ListArmPorts`/`SetArmConnection` services referenced in `arm_node_base.py` — that interfaces package isn't included in this folder, so ask if it's missing.
- Hardware use needs the `dynamixel-sdk` pip package (`pip install dynamixel-sdk`); it's only imported when you actually call `connect()`, so the sim node works without it.
- `ports.py` only looks for Linux-style serial device paths — consistent with the main README's advice to run the control code on Ubuntu (or Windows) rather than Mac.
- All positions/speeds in `/arm/command` are in radians and the geometry in `kinematics.py` (link lengths, joint-4 limits) matches the physical OpenManipulator-X; if your unit's dimensions differ, adjust `ArmGeometry` accordingly.

- need to `pip install dynamixel-sdk`!!!