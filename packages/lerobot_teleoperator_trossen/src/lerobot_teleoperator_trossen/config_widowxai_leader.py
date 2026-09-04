from dataclasses import dataclass, field

import numpy as np
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("widowxai_leader_teleop")
@dataclass
class WidowXAILeaderTeleopConfig(TeleoperatorConfig):
    ip_address: str
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
    staged_positions: list[float] = field(
        default_factory=lambda: [0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0, 0]
    )
