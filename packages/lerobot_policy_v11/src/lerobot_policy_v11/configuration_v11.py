from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("v11")
@dataclass
class V11Config(PreTrainedConfig):
    """Inference contract for a self-contained V11 end-to-end checkpoint."""

    # Retained only so older exported configs remain loadable. The integrated
    # runtime constructs the architecture locally and needs no source checkout.
    backbone_checkpoint: str = ""
    backbone_source_root: str = ""
    tcc_real_robot_source_root: str = ""
    backbone_family: str = "ours_rn50"
    freeze_vision_backbone: bool = True
    backbone_image_size: int = 224
    backbone_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    backbone_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    feature_dim: int = 2048
    hidden_dimensions: tuple[int, ...] = (256, 256)
    action_dim: int = 7
    action_chunk_size: int = 40
    n_action_steps: int = 10
    proprioception_dim: int = 7
    number_of_tasks: int = 1
    task_index: int = 0
    action_feature_names: list[str] = field(
        default_factory=lambda: [
            "joint_0.pos",
            "joint_1.pos",
            "joint_2.pos",
            "joint_3.pos",
            "joint_4.pos",
            "joint_5.pos",
            "left_carriage_joint.pos",
        ]
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backbone_family != "ours_rn50":
            raise ValueError("The V11 official-runtime adapter supports ours_rn50 only")
        if not self.freeze_vision_backbone:
            raise ValueError("V11 was trained with a frozen vision backbone")
        if self.feature_dim <= 0 or self.action_dim != 7:
            raise ValueError("V11 requires positive RN50 features and seven actions")
        if not self.hidden_dimensions or any(x <= 0 for x in self.hidden_dimensions):
            raise ValueError("V11 hidden dimensions must be positive")
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        if not 1 <= self.n_action_steps <= self.action_chunk_size:
            raise ValueError("n_action_steps must be within the action chunk")
        if self.proprioception_dim != 7 or self.number_of_tasks != 1:
            raise ValueError("V11 expects one task and seven-dimensional state")
        if self.task_index != 0:
            raise ValueError("The single-task V11 adapter requires task_index=0")
        if len(self.action_feature_names) != self.action_dim:
            raise ValueError("action_feature_names must match the action dimension")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-3, weight_decay=0.0)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if len(self.image_features) != 1:
            raise ValueError("V11 requires exactly one policy image feature")
        if "observation.state" not in self.input_features:
            raise ValueError("V11 requires observation.state")
        if "action" not in self.output_features:
            raise ValueError("V11 requires an action output feature")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.action_chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
