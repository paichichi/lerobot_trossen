from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_rn50_late")
@dataclass
class ACTRN50LateConfig(ACTConfig):
    """Official ACT RN50 with only layer3 and layer4 fine-tuned."""

    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = "ResNet50_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-6
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.vision_backbone != "resnet50":
            raise ValueError("act_rn50_late requires vision_backbone='resnet50'")
        if self.replace_final_stride_with_dilation:
            raise ValueError(
                "act_rn50_late keeps the official RN50 stride unchanged so freezing "
                "the early backbone is the only optimization variable"
            )
        if self.optimizer_lr <= 0 or self.optimizer_lr_backbone <= 0:
            raise ValueError("Optimizer learning rates must be positive")
