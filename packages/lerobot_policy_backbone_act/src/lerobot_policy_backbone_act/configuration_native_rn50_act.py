from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim.schedulers import CosineAnnealingWithWarmupSchedulerConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_rn50_full")
@dataclass
class ACTRN50FullConfig(ACTConfig):
    """Full-resolution ACT designed around the trained ours RN50."""

    backbone_checkpoint: str = ""
    backbone_source_root: str = ""
    backbone_family: str = "ours_rn50"
    freeze_vision_backbone: bool = False
    backbone_image_height: int = 480
    backbone_image_width: int = 640
    backbone_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    backbone_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = None
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 4
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-6
    optimizer_weight_decay: float = 1e-4
    scheduler_warmup_steps: int = 1000

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        native_contract = {
            "backbone_family": (self.backbone_family, "ours_rn50"),
            "freeze_vision_backbone": (self.freeze_vision_backbone, False),
            "backbone_image_height": (self.backbone_image_height, 480),
            "backbone_image_width": (self.backbone_image_width, 640),
            "vision_backbone": (self.vision_backbone, "resnet50"),
            "pretrained_backbone_weights": (self.pretrained_backbone_weights, None),
        }
        mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in native_contract.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "act_rn50_full requires the native trainable ours RN50 contract: "
                + ", ".join(mismatches)
            )
        if self.normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
            raise ValueError(
                "act_rn50_full requires VISUAL=IDENTITY; the RN50 applies its "
                "ImageNet normalization internally"
            )
        if len(self.backbone_image_mean) != 3 or len(self.backbone_image_std) != 3:
            raise ValueError("Backbone image mean/std must contain three RGB values")
        if any(value <= 0 for value in self.backbone_image_std):
            raise ValueError("Backbone image std values must be positive")
        if self.scheduler_warmup_steps < 0:
            raise ValueError("scheduler_warmup_steps must be non-negative")

    def get_scheduler_preset(self) -> CosineAnnealingWithWarmupSchedulerConfig:
        return CosineAnnealingWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps
        )


# Load checkpoints created before the ACT-RN50-full name was finalized.
NativeRN50ACTConfig = ACTRN50FullConfig
