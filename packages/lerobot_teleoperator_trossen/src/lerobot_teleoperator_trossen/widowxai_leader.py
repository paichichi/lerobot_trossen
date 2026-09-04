import logging
import time

import trossen_arm
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_widowxai_leader import WidowXAILeaderTeleopConfig

logger = logging.getLogger(__name__)


class WidowXAILeaderTeleop(Teleoperator):
    config_class = WidowXAILeaderTeleopConfig
    name = "widowxai_leader_teleop"

    def __init__(self, config: WidowXAILeaderTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self.driver = trossen_arm.TrossenArmDriver()

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in self.config.joint_names}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

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
            end_effector=trossen_arm.StandardEndEffector.wxai_v0_leader,
            serv_ip=self.config.ip_address,
            clear_error=True,
        )
        self.configure()
        logger.info("Leader connected")

    def configure(self) -> None:
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self.driver.set_all_positions(
            self.config.staged_positions, goal_time=2.0, blocking=True
        )
        self.driver.set_all_modes(trossen_arm.Mode.external_effort)
        self.driver.set_all_external_efforts(
            [0.0] * len(self.config.joint_names), goal_time=0.0, blocking=True
        )

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        positions = self.driver.get_all_positions()
        action = dict(
            zip(
                (f"{name}.pos" for name in self.config.joint_names),
                positions,
                strict=True,
            )
        )
        logger.debug("Leader read action: %.1fms", (time.perf_counter() - start) * 1e3)
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        self.driver.set_all_modes(trossen_arm.Mode.position)
        self.driver.set_all_positions(
            self.config.staged_positions, goal_time=2.0, blocking=True
        )
        self.driver.set_all_positions(
            [0.0] * len(self.config.joint_names), goal_time=2.0, blocking=True
        )
        self.driver.cleanup()
        logger.info("Leader disconnected")
