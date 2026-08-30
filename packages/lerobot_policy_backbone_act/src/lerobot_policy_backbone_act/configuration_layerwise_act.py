from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_rn50_layerwise")
@dataclass
class ACTRN50LayerwiseConfig(ACTConfig):
    """Official ACT RN50 with a small, three-group fine-tuning preset."""

    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = "ResNet50_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False

    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 3e-6
    optimizer_lr_backbone_early: float = 3e-7
    optimizer_weight_decay: float = 1e-4

    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 8000
    scheduler_decay_ratio: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.vision_backbone != "resnet50":
            raise ValueError("act_rn50_layerwise requires vision_backbone='resnet50'")
        if self.replace_final_stride_with_dilation:
            raise ValueError(
                "act_rn50_layerwise keeps the official RN50 stride unchanged so the "
                "fine-tuning strategy remains the only architecture-side variable"
            )
        learning_rates = {
            "optimizer_lr": self.optimizer_lr,
            "optimizer_lr_backbone": self.optimizer_lr_backbone,
            "optimizer_lr_backbone_early": self.optimizer_lr_backbone_early,
        }
        invalid = [name for name, value in learning_rates.items() if value <= 0]
        if invalid:
            raise ValueError(f"Learning rates must be positive: {', '.join(invalid)}")
        if self.scheduler_warmup_steps < 0:
            raise ValueError("scheduler_warmup_steps must be non-negative")
        if self.scheduler_decay_steps <= self.scheduler_warmup_steps:
            raise ValueError(
                "scheduler_decay_steps must be greater than scheduler_warmup_steps"
            )
        if not 0 < self.scheduler_decay_ratio <= 1:
            raise ValueError("scheduler_decay_ratio must be in (0, 1]")

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.optimizer_lr * self.scheduler_decay_ratio,
        )
