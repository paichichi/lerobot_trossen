import logging
import time
from typing import Any

import trossen_arm
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_widowxai_follower import WidowXAIFollowerConfig

logger = logging.getLogger(__name__)


class WidowXAIFollower(Robot):
    config_class = WidowXAIFollowerConfig
    name = "widowxai_follower_robot"

    def __init__(self, config: WidowXAIFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.driver = trossen_arm.TrossenArmDriver()
        self.min_time_to_move = config.min_time_to_move_multiplier / config.loop_rate

    @property
    def observation_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.config.joint_names}

    @property
    def action_features(self) -> dict[str, type]:
        return self.observation_features

    @property
    def is_connected(self) -> bool:
        return self.driver.get_is_configured()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.driver.configure(
            model=trossen_arm.Model.wxai_v0,
            end_effector=trossen_arm.StandardEndEffector.wxai_v0_follower,
            serv_ip=self.config.ip_address,
            clear_error=True,
        )
        self.configure()
        logger.info("Follower connected")

    def configure(self) -> None:
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self.driver.set_all_positions(
            self.config.staged_positions, goal_time=2.0, blocking=True
        )

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        start = time.perf_counter()
        positions = self.driver.get_robot_output().joint.all.positions
        observation = dict(
            zip(
                (f"{name}.pos" for name in self.config.joint_names),
                positions,
                strict=True,
            )
        )
        logger.debug("Follower read state: %.1fms", (time.perf_counter() - start) * 1e3)
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        goals = {
            key.removesuffix(".pos"): value
            for key, value in action.items()
            if key.endswith(".pos")
        }
        if self.config.max_relative_target is not None:
            current = dict(
                zip(self.config.joint_names, self.driver.get_all_positions(), strict=True)
            )
            goals = ensure_safe_goal_position(
                {name: (goal, current[name]) for name, goal in goals.items()},
                self.config.max_relative_target,
            )
        self.driver.set_all_positions(
            [goals[name] for name in self.config.joint_names],
            goal_time=self.min_time_to_move,
            blocking=False,
        )
        return {f"{name}.pos": value for name, value in goals.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        self.driver.set_all_positions(
            self.config.staged_positions, goal_time=2.0, blocking=True
        )
        self.driver.set_all_positions(
            [0.0] * len(self.config.joint_names), goal_time=2.0, blocking=True
        )
        self.driver.cleanup()
        logger.info("Follower disconnected")
