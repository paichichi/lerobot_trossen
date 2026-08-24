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

    # Upstream ACT uses this only to allocate its 2048-channel image projection.
    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backbone_family != "ours_rn50":
            raise ValueError(
                "The first backbone_act lite release supports ours_rn50 only; "
                "ViT requires a 768-channel patch-token spatial adapter"
            )
        if self.vision_backbone != "resnet50":
            raise ValueError("ours_rn50 must use ACT's ResNet50-shaped projection")
        if self.backbone_image_size <= 0:
            raise ValueError("backbone_image_size must be positive")
        if len(self.backbone_image_mean) != 3 or len(self.backbone_image_std) != 3:
            raise ValueError("Backbone image mean/std must contain three RGB values")
        if any(value <= 0 for value in self.backbone_image_std):
            raise ValueError("Backbone image std values must be positive")
        if self.normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
            raise ValueError(
                "backbone_act requires VISUAL=IDENTITY to prevent dataset and "
                "ImageNet normalization from being applied twice"
            )
