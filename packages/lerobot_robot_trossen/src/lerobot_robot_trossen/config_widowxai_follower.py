from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("widowxai_follower_robot")
@dataclass
class WidowXAIFollowerConfig(RobotConfig):
    # IP address of the arm
    ip_address: str

    # `max_relative_target` limits the magnitude of the relative positional target vector for
    # safety purposes. Set this to a positive scalar for the same value on every motor, or a
    # dictionary keyed by joint name when arm joints (rad) and the gripper (m) need different caps.
    max_relative_target: float | dict[str, float] | None = 5.0

    # Multiplier for computing minimum time (in seconds) for the arm to reach a target position.
    # The final goal time is computed as: min_time_to_move = multiplier / fps.
    # A smaller multiplier results in faster (but potentially jerky) motion.
    # A larger multiplier results in smoother motion but with increased lag.
    # A recommended starting value is 3.0.
    min_time_to_move_multiplier: float = 3.0

    # Optional time-aware smoothing for absolute joint-position policies. These
    # limits run before max_relative_target, which remains the final safety cap.
    arm_max_velocity_rad_s: float | None = None
    arm_max_acceleration_rad_s2: float | None = None
    gripper_max_velocity_m_s: float | None = None
    gripper_deadband_m: float = 0.0
    postprocess_max_dt_multiplier: float = 2.0

    # Optional dataset-home staging before the first rollout observation.
    # Unlike staged_positions, this pose is also verified and becomes the
    # initial position captured by LeRobot for episode resets and shutdown.
    startup_home_positions: list[float] | None = None
    startup_home_goal_time_s: float = 10.0
    startup_home_settle_time_s: float = 1.0
    startup_home_max_arm_error_rad: float = 0.03
    startup_home_max_gripper_error_m: float = 0.002

    # The controller can briefly retain a stale TCP session after an aborted
    # process. Retry only transient transport failures, before any home motion.
    controller_connect_attempts: int = 3
    controller_connect_retry_delay_s: float = 2.0

    # Best-effort safe shutdown. Once the controller is configured, every
    # normal or exceptional disconnect stages the arm and then folds it.
    fold_on_disconnect: bool = True
    fold_staging_goal_time_s: float = 3.0
    folded_positions: list[float] = field(default_factory=lambda: [0.0] * 7)
    fold_goal_time_s: float = 3.0
    fold_max_arm_error_rad: float = 0.12
    fold_max_gripper_error_m: float = 0.003

    # Control loop rate in Hz
    loop_rate: int = 30

    # Include per-joint velocity (`.vel`) in observations.
    include_velocity: bool = False

    # Include per-joint effort (`.eff`) in observations. This is the total motor effort, combining
    # gravity, friction, and any external load. Measured in Nm for the arm joints and N for the
    # gripper carriage. Nonzero even when the arm is holding still against gravity.
    include_effort: bool = False

    # Include per-joint external effort (`.ext_eff`) in observations. This is the estimated
    # externally applied effort, after gravity and friction compensation. Measured in Nm for the
    # arm joints and N for the gripper carriage. Useful for contact and force sensing; an unloaded
    # arm reports values near zero.
    include_external_effort: bool = False

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.

    # Joint names for the WidowX AI follower arm
    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "left_carriage_joint",
        ]
    )

    # "Staged" positions in rad for the arm and m for the gripper
    #
    # The robot will move to these positions when first started and before the arm is sent to the
    # sleep position.
    staged_positions: list[float] = field(
        default_factory=lambda: [0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0]
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        limits = (
            self.arm_max_velocity_rad_s,
            self.arm_max_acceleration_rad_s2,
            self.gripper_max_velocity_m_s,
        )
        enabled = [value is not None for value in limits]
        if any(enabled) and not all(enabled):
            raise ValueError("All stable postprocess velocity/acceleration limits are required")
        if any(value is not None and value <= 0 for value in limits):
            raise ValueError("Stable postprocess limits must be positive")
        if self.gripper_deadband_m < 0:
            raise ValueError("gripper_deadband_m must be non-negative")
        if self.postprocess_max_dt_multiplier < 1.0:
            raise ValueError("postprocess_max_dt_multiplier must be at least one")
        if self.startup_home_positions is not None:
            if len(self.startup_home_positions) != len(self.joint_names):
                raise ValueError("startup_home_positions must contain one value per joint")
            if not all(isfinite(value) for value in self.startup_home_positions):
                raise ValueError("startup_home_positions must contain only finite values")
        if self.startup_home_goal_time_s <= 0:
            raise ValueError("startup_home_goal_time_s must be positive")
        if self.startup_home_settle_time_s < 0:
            raise ValueError("startup_home_settle_time_s must be non-negative")
        if self.startup_home_max_arm_error_rad <= 0:
            raise ValueError("startup_home_max_arm_error_rad must be positive")
        if self.startup_home_max_gripper_error_m <= 0:
            raise ValueError("startup_home_max_gripper_error_m must be positive")
        if self.controller_connect_attempts < 1:
            raise ValueError("controller_connect_attempts must be at least one")
        if self.controller_connect_retry_delay_s < 0:
            raise ValueError("controller_connect_retry_delay_s must be non-negative")
        if self.fold_staging_goal_time_s <= 0:
            raise ValueError("fold_staging_goal_time_s must be positive")
        if len(self.folded_positions) != len(self.joint_names):
            raise ValueError("folded_positions must contain one value per joint")
        if not all(isfinite(value) for value in self.folded_positions):
            raise ValueError("folded_positions must contain only finite values")
        if self.fold_goal_time_s <= 0:
            raise ValueError("fold_goal_time_s must be positive")
        if self.fold_max_arm_error_rad <= 0:
            raise ValueError("fold_max_arm_error_rad must be positive")
        if self.fold_max_gripper_error_m <= 0:
            raise ValueError("fold_max_gripper_error_m must be positive")
