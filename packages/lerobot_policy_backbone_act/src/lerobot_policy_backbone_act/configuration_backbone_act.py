from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("backbone_act")
@dataclass
class BackboneACTConfig(ACTConfig):
    """Official ACT with one frozen ViT-B/16 image encoder."""

    backbone_checkpoint: str = ""
    backbone_source_root: str = ""
    backbone_family: str = "ours_vit"
    backbone_image_size: int = 224
    backbone_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    backbone_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    freeze_vision_backbone: bool = True

    # Retained only so checkpoints created during the earlier experiments still
    # deserialize. The clean pipeline accepts frozen backbones only.
    vit_trainable_last_blocks: int | None = None
    vit_train_layer_norm_only: bool = False

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backbone_family not in {"ours_vit", "pretrained_vit"}:
            raise ValueError("backbone_family must be 'ours_vit' or 'pretrained_vit'")
        if not self.freeze_vision_backbone:
            raise ValueError("The clean backbone_act pipeline supports frozen ViT backbones only")
        if self.vit_trainable_last_blocks is not None or self.vit_train_layer_norm_only:
            raise ValueError("Selective ViT fine-tuning is no longer supported")
        if self.backbone_image_size != 224:
            raise ValueError("ViT-B/16 checkpoints require 224x224 images")
        if self.vision_backbone != "resnet50":
            raise ValueError("vision_backbone must remain resnet50 for ACT initialization")
        if len(self.backbone_image_mean) != 3 or len(self.backbone_image_std) != 3:
            raise ValueError("Image mean and std must each contain three RGB values")
        if any(value <= 0 for value in self.backbone_image_std):
            raise ValueError("Image standard deviations must be positive")
        if self.normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
            raise ValueError("Visual normalization is owned by the ViT adapter")
