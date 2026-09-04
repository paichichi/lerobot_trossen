from dataclasses import dataclass, field

import numpy as np
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("widowxai_follower_robot")
@dataclass
class WidowXAIFollowerConfig(RobotConfig):
    ip_address: str
    max_relative_target: float | None = 0.07
    min_time_to_move_multiplier: float = 1.5
    loop_rate: int = 20
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
