from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("backbone_act")
@dataclass
class BackboneACTConfig(ACTConfig):
    """Official ACT whose spatial tokens come from a trained upstream backbone.

    State and action normalization remain the official ACT implementation.
    Visual normalization is owned by the backbone adapter so that training and
    robot inference exactly reproduce the upstream representation contract.
    """

    backbone_checkpoint: str = ""
    backbone_source_root: str = ""
    freeze_vision_backbone: bool = True
    # None preserves the existing all-or-nothing behavior. When the backbone is
    # trainable, a positive value restricts ViT tuning to the final N encoder
    # blocks plus the encoder's final LayerNorm.
    vit_trainable_last_blocks: int | None = None
    # Train only the affine weight and bias of every ViT LayerNorm. The ACT
    # policy remains normally trainable; every non-LN ViT parameter is frozen.
    vit_train_layer_norm_only: bool = False
    backbone_family: str = "ours_rn50"
    backbone_image_size: int = 224
    backbone_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    backbone_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Dataset state/action statistics remain active. Visual input is converted
    # to [0, 1], resized, and ImageNet-normalized inside the backbone adapter.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Upstream ACT uses this only to bootstrap its image modules. The policy
    # replaces both the backbone and projection for ours_vit after construction.
    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backbone_family not in {"ours_rn50", "ours_vit", "pretrained_vit"}:
            raise ValueError(
                "backbone_family must be one of "
                "{'ours_rn50', 'ours_vit', 'pretrained_vit'}"
            )
        if self.vision_backbone != "resnet50":
            raise ValueError(
                "backbone_act uses vision_backbone='resnet50' while bootstrapping "
                "the official ACT image modules"
            )
        if self.backbone_image_size <= 0:
            raise ValueError("backbone_image_size must be positive")
        if self.backbone_family in {"ours_vit", "pretrained_vit"} and self.backbone_image_size != 224:
            raise ValueError(
                "ViT-B/16 checkpoints require a fixed 224x224 positional embedding"
            )
        if self.vit_trainable_last_blocks is not None:
            if self.backbone_family not in {"ours_vit", "pretrained_vit"}:
                raise ValueError(
                    "vit_trainable_last_blocks is only valid for a ViT backbone"
                )
            if self.freeze_vision_backbone:
                raise ValueError(
                    "vit_trainable_last_blocks requires freeze_vision_backbone=false"
                )
            if not 1 <= self.vit_trainable_last_blocks <= 12:
                raise ValueError("vit_trainable_last_blocks must be between 1 and 12")
        if self.vit_train_layer_norm_only:
            if self.backbone_family not in {"ours_vit", "pretrained_vit"}:
                raise ValueError(
                    "vit_train_layer_norm_only is only valid for a ViT backbone"
                )
            if self.freeze_vision_backbone:
                raise ValueError(
                    "vit_train_layer_norm_only requires "
                    "freeze_vision_backbone=false"
                )
            if self.vit_trainable_last_blocks is not None:
                raise ValueError(
                    "vit_train_layer_norm_only and vit_trainable_last_blocks "
                    "are mutually exclusive"
                )
        if len(self.backbone_image_mean) != 3 or len(self.backbone_image_std) != 3:
            raise ValueError("Backbone image mean/std must contain three RGB values")
        if any(value <= 0 for value in self.backbone_image_std):
            raise ValueError("Backbone image std values must be positive")
        if self.normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
            raise ValueError(
                "backbone_act requires VISUAL=IDENTITY to prevent dataset and "
                "ImageNet normalization from being applied twice"
            )
