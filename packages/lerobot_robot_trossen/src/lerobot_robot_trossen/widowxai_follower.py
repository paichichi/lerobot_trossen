import logging
import time
from typing import Any

import trossen_arm
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
from lerobot_robot_trossen.stable_postprocess import TimeAwareJointTargetFilter

logger = logging.getLogger(__name__)


class WidowXAIFollower(Robot):
    """
    [WidowX AI](https://www.trossenrobotics.com/widowx-ai) by Trossen Robotics
    """

    config_class = WidowXAIFollowerConfig
    name = "widowxai_follower_robot"

    def __init__(self, config: WidowXAIFollowerConfig):
        super().__init__(config)
        self.config = config

        self.driver = trossen_arm.TrossenArmDriver()
        self.cameras = make_cameras_from_configs(config.cameras)
        self.min_time_to_move = (
            config.min_time_to_move_multiplier / self.config.loop_rate
        )
        self.action_filter = None
        if config.arm_max_velocity_rad_s is not None:
            assert config.arm_max_acceleration_rad_s2 is not None
            assert config.gripper_max_velocity_m_s is not None
            self.action_filter = TimeAwareJointTargetFilter(
                arm_joint_names=tuple(config.joint_names[:-1]),
                gripper_joint_name=config.joint_names[-1],
                nominal_dt=1.0 / config.loop_rate,
                max_dt_multiplier=config.postprocess_max_dt_multiplier,
                arm_max_velocity_rad_s=config.arm_max_velocity_rad_s,
                arm_max_acceleration_rad_s2=config.arm_max_acceleration_rad_s2,
                gripper_max_velocity_m_s=config.gripper_max_velocity_m_s,
                gripper_deadband_m=config.gripper_deadband_m,
            )

    @property
    def _joint_ft(self) -> dict[str, type]:
        return {f"{joint_name}.pos": float for joint_name in self.config.joint_names}

    @property
    def _joint_vel_ft(self) -> dict[str, type]:
        return {f"{joint_name}.vel": float for joint_name in self.config.joint_names}

    @property
    def _joint_eff_ft(self) -> dict[str, type]:
        return {f"{joint_name}.eff": float for joint_name in self.config.joint_names}

    @property
    def _joint_ext_eff_ft(self) -> dict[str, type]:
        return {
            f"{joint_name}.ext_eff": float for joint_name in self.config.joint_names
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # Depth cameras add a separate (H, W, 1) "<cam>_depth" feature; single channel flags it as depth.
        features: dict[str, tuple] = {}
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                features[cam_key] = (cam.height, cam.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{cam_key}_depth"] = (cam.height, cam.width, 1)
        return features

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        joint_ft = {**self._joint_ft}
        if self.config.include_velocity:
            joint_ft.update(self._joint_vel_ft)
        if self.config.include_effort:
            joint_ft.update(self._joint_eff_ft)
        if self.config.include_external_effort:
            joint_ft.update(self._joint_ext_eff_ft)
        return {**joint_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        return self._joint_ft

    @property
    def is_connected(self) -> bool:
        return self.driver.get_is_configured() and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.driver.configure(
            model=trossen_arm.Model.wxai_v0,
            end_effector=trossen_arm.StandardEndEffector.wxai_v0_follower,
            serv_ip=self.config.ip_address,
            clear_error=True,
        )
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Trossen Arm robots do not require calibration
        return True

    def calibrate(self) -> None:
        # Trossen Arm robots do not require calibration
        pass

    def configure(self) -> None:
        # Set the arm to position control mode
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self.driver.set_all_positions(
            self.config.staged_positions,
            goal_time=2.0,
            blocking=True,
        )
        if self.action_filter is not None:
            self.action_filter.reset()
            logger.info(
                "Stable joint-target postprocess enabled: arm %.3f rad/s, "
                "arm %.3f rad/s^2, gripper %.3f m/s, deadband %.4f m",
                self.config.arm_max_velocity_rad_s,
                self.config.arm_max_acceleration_rad_s2,
                self.config.gripper_max_velocity_m_s,
                self.config.gripper_deadband_m,
            )

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Create observation dictionary with joint positions, plus velocities and efforts when
        # enabled via include_velocity / include_effort / include_external_effort.
        start = time.perf_counter()

        robot_all_joint_outputs = self.driver.get_robot_output().joint.all
        obs_dict = {}
        obs_dict.update(
            {
                f"{joint_name}.pos": pos
                for joint_name, pos in zip(
                    self.config.joint_names,
                    robot_all_joint_outputs.positions,
                    strict=True,
                )
            }
        )
        if self.config.include_velocity:
            obs_dict.update(
                {
                    f"{joint_name}.vel": vel
                    for joint_name, vel in zip(
                        self.config.joint_names,
                        robot_all_joint_outputs.velocities,
                        strict=True,
                    )
                }
            )
        if self.config.include_effort:
            obs_dict.update(
                {
                    f"{joint_name}.eff": eff
                    for joint_name, eff in zip(
                        self.config.joint_names,
                        robot_all_joint_outputs.efforts,
                        strict=True,
                    )
                }
            )
        if self.config.include_external_effort:
            obs_dict.update(
                {
                    f"{joint_name}.ext_eff": ext_eff
                    for joint_name, ext_eff in zip(
                        self.config.joint_names,
                        robot_all_joint_outputs.external_efforts,
                        strict=True,
                    )
                }
            )

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # read_latest*() is a non-blocking peek of the latest frame; depth cameras also yield "<cam>_depth".
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        # Read the present joint state once for both the stable filter and final
        # hard safety cap. The policy action remains an absolute joint target.
        if self.action_filter is not None or self.config.max_relative_target is not None:
            present_pos = dict(
                zip(
                    self.config.joint_names,
                    self.driver.get_all_positions(),
                    strict=True,
                )
            )
        if self.action_filter is not None:
            goal_pos = self.action_filter.apply(
                goal_pos,
                present_pos,
                now=time.perf_counter(),
            )

        # Cap any remaining large relative goal as the final safety layer.
        if self.config.max_relative_target is not None:
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        # Send goal position to the arm
        self.driver.set_all_positions(
            goal_positions=[
                goal_pos[joint_name] for joint_name in self.config.joint_names
            ],
            goal_time=self.min_time_to_move,
            blocking=False,
        )
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Move the arm to the staged positions before disconnecting
        self.driver.set_all_positions(
            self.config.staged_positions,
            goal_time=2.0,
            blocking=True,
        )
        # Move the arm to the sleep position (all positions to 0.0)
        self.driver.set_all_positions(
            [0.0] * len(self.config.joint_names),
            goal_time=2.0,
            blocking=True,
        )

        self.driver.cleanup()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
