import logging
import time
from contextlib import suppress
from typing import Any

import trossen_arm
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from lerobot_robot_trossen.config_widowxai_follower import WidowXAIFollowerConfig
from lerobot_robot_trossen.stable_postprocess import (
    TimeAwareJointTargetFilter,
    home_tracking_errors,
    is_transient_controller_transport_error,
)

logger = logging.getLogger(__name__)

CAMERA_RECOVERABLE_ERRORS = (
    TimeoutError,
    RuntimeError,
    OSError,
    DeviceAlreadyConnectedError,
    DeviceNotConnectedError,
)


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
        self._driver_configured = False
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
        # Treat an active controller as connected even if a camera drops. This
        # ensures LeRobot teardown still calls disconnect() and folds the arm.
        return self._driver_configured and self.driver.get_is_configured()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for attempt in range(1, self.config.controller_connect_attempts + 1):
            try:
                self.driver.configure(
                    model=trossen_arm.Model.wxai_v0,
                    end_effector=trossen_arm.StandardEndEffector.wxai_v0_follower,
                    serv_ip=self.config.ip_address,
                    clear_error=True,
                )
                self._driver_configured = True
                break
            except RuntimeError as error:
                retry = (
                    attempt < self.config.controller_connect_attempts
                    and is_transient_controller_transport_error(error)
                )
                with suppress(Exception):
                    self.driver.cleanup()
                self._driver_configured = False
                if not retry:
                    raise RuntimeError(
                        "Arm controller connection failed before dataset-home staging; "
                        "no home command or policy action was sent."
                    ) from error
                logger.warning(
                    "Transient arm-controller connection failure (%d/%d); "
                    "rebuilding the driver and retrying in %.1f seconds: %s",
                    attempt,
                    self.config.controller_connect_attempts,
                    self.config.controller_connect_retry_delay_s,
                    error,
                )
                time.sleep(self.config.controller_connect_retry_delay_s)
                self.driver = trossen_arm.TrossenArmDriver()
        try:
            if not self.is_calibrated and calibrate:
                self.calibrate()

            for cam in self.cameras.values():
                cam.connect()

            self.configure()
        except Exception:
            # Once controller configuration succeeds, never leave an arm
            # active merely because camera setup or home verification failed.
            with suppress(Exception):
                self._shutdown_hardware(fold=self.config.fold_on_disconnect)
            raise
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
        startup_positions = (
            self.config.startup_home_positions
            if self.config.startup_home_positions is not None
            else self.config.staged_positions
        )
        startup_goal_time = (
            self.config.startup_home_goal_time_s
            if self.config.startup_home_positions is not None
            else 2.0
        )
        if self.config.startup_home_positions is not None:
            logger.info(
                "Moving to verified dataset home over %.1f seconds before rollout",
                startup_goal_time,
            )
        self.driver.set_all_positions(
            startup_positions,
            goal_time=startup_goal_time,
            blocking=True,
        )
        if self.config.startup_home_positions is not None:
            time.sleep(self.config.startup_home_settle_time_s)
            observed = [float(value) for value in self.driver.get_all_positions()]
            arm_error, gripper_error = home_tracking_errors(
                self.config.startup_home_positions,
                observed,
            )
            if arm_error > self.config.startup_home_max_arm_error_rad:
                raise RuntimeError(
                    f"Dataset-home arm error {arm_error:.6f} rad exceeds "
                    f"{self.config.startup_home_max_arm_error_rad:.6f} rad"
                )
            if gripper_error > self.config.startup_home_max_gripper_error_m:
                raise RuntimeError(
                    f"Dataset-home gripper error {gripper_error:.6f} m exceeds "
                    f"{self.config.startup_home_max_gripper_error_m:.6f} m"
                )
            logger.info(
                "Verified dataset home: arm error %.6f rad, gripper error %.6f m",
                arm_error,
                gripper_error,
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
                obs_dict[cam_key] = self._read_camera_with_recovery(
                    cam_key,
                    cam,
                    depth=False,
                )
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = self._read_camera_with_recovery(
                    cam_key,
                    cam,
                    depth=True,
                )
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    def _read_camera_with_recovery(
        self,
        cam_key: str,
        cam: Any,
        *,
        depth: bool,
    ) -> Any:
        """Read a fresh frame, restarting only a failed camera when necessary.

        No stale or synthetic image is sent to the policy. While this method is
        reconnecting, no new arm action is issued, so the controller holds the
        previously commanded target. If recovery is exhausted, the exception
        propagates and the normal rollout teardown safely folds the arm.
        """

        stream_name = "depth" if depth else "RGB"

        def read_frame() -> Any:
            reader = cam.read_latest_depth if depth else cam.read_latest
            return reader(max_age_ms=self.config.camera_frame_max_age_ms)

        try:
            return read_frame()
        except CAMERA_RECOVERABLE_ERRORS as error:
            last_error: Exception = error
            logger.warning(
                "%s camera %s stream failed; holding the last arm target and "
                "restarting only this camera: %s",
                cam_key,
                stream_name,
                error,
            )

        for attempt in range(1, self.config.camera_reconnect_attempts + 1):
            with suppress(Exception):
                cam.disconnect()
            if self.config.camera_reconnect_delay_s:
                time.sleep(self.config.camera_reconnect_delay_s)
            try:
                cam.connect()
                frame = read_frame()
                logger.info(
                    "%s camera recovered on attempt %d/%d",
                    cam_key,
                    attempt,
                    self.config.camera_reconnect_attempts,
                )
                return frame
            except CAMERA_RECOVERABLE_ERRORS as error:
                last_error = error
                logger.warning(
                    "%s camera reconnect attempt %d/%d failed: %s",
                    cam_key,
                    attempt,
                    self.config.camera_reconnect_attempts,
                    error,
                )

        raise RuntimeError(
            f"{cam_key} camera {stream_name} stream could not recover after "
            f"{self.config.camera_reconnect_attempts} attempt(s)"
        ) from last_error

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

    def _shutdown_hardware(self, *, fold: bool) -> None:
        fold_error: Exception | None = None
        if self._driver_configured and fold:
            try:
                logger.info(
                    "Safe shutdown: moving through staged waypoint over %.1f seconds",
                    self.config.fold_staging_goal_time_s,
                )
                self.driver.set_all_modes(trossen_arm.Mode.position)
                self.driver.set_all_positions(
                    self.config.staged_positions,
                    goal_time=self.config.fold_staging_goal_time_s,
                    blocking=True,
                )
                logger.info(
                    "Safe shutdown: moving to folded pose over %.1f seconds",
                    self.config.fold_goal_time_s,
                )
                self.driver.set_all_positions(
                    self.config.folded_positions,
                    goal_time=self.config.fold_goal_time_s,
                    blocking=True,
                )
                observed = [float(value) for value in self.driver.get_all_positions()]
                arm_error, gripper_error = home_tracking_errors(
                    self.config.folded_positions,
                    observed,
                )
                if arm_error > self.config.fold_max_arm_error_rad:
                    raise RuntimeError(
                        f"Folded-pose arm error {arm_error:.6f} rad exceeds "
                        f"{self.config.fold_max_arm_error_rad:.6f} rad"
                    )
                if gripper_error > self.config.fold_max_gripper_error_m:
                    raise RuntimeError(
                        f"Folded-pose gripper error {gripper_error:.6f} m exceeds "
                        f"{self.config.fold_max_gripper_error_m:.6f} m"
                    )
                logger.info(
                    "Safe shutdown: folded pose verified (arm %.6f rad, gripper %.6f m)",
                    arm_error,
                    gripper_error,
                )
            except Exception as error:
                fold_error = error
                logger.exception("Safe shutdown could not reach the folded pose")

        if self._driver_configured:
            with suppress(Exception):
                self.driver.cleanup()
            self._driver_configured = False
        for cam in self.cameras.values():
            if cam.is_connected:
                with suppress(Exception):
                    cam.disconnect()

        if fold_error is not None:
            raise RuntimeError(
                "Controller was cleaned up, but the arm could not be confirmed at the folded pose"
            ) from fold_error

    def disconnect(self):
        if not self._driver_configured and not any(
            cam.is_connected for cam in self.cameras.values()
        ):
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._shutdown_hardware(fold=self.config.fold_on_disconnect)

        logger.info(f"{self} disconnected.")
