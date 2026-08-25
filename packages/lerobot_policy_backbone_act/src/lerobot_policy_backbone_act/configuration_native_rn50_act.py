from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("native_rn50_act")
@dataclass
class NativeRN50ACTConfig(ACTConfig):
    """Full official ACT with the trained ours RN50 at native 480x640 resolution."""

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

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        required = {
            "backbone_family": (self.backbone_family, "ours_rn50"),
            "freeze_vision_backbone": (self.freeze_vision_backbone, False),
            "backbone_image_height": (self.backbone_image_height, 480),
            "backbone_image_width": (self.backbone_image_width, 640),
            "vision_backbone": (self.vision_backbone, "resnet50"),
            "pretrained_backbone_weights": (self.pretrained_backbone_weights, None),
            "dim_model": (self.dim_model, 512),
            "n_heads": (self.n_heads, 8),
            "dim_feedforward": (self.dim_feedforward, 3200),
            "n_encoder_layers": (self.n_encoder_layers, 4),
            "n_decoder_layers": (self.n_decoder_layers, 1),
            "n_vae_encoder_layers": (self.n_vae_encoder_layers, 4),
        }
        mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "native_rn50_act is locked to native 480x640 and full official ACT: "
                + ", ".join(mismatches)
            )
        if self.normalization_mapping.get("VISUAL") != NormalizationMode.IDENTITY:
            raise ValueError(
                "native_rn50_act requires VISUAL=IDENTITY; the RN50 applies its "
                "ImageNet normalization internally"
            )
        if len(self.backbone_image_mean) != 3 or len(self.backbone_image_std) != 3:
            raise ValueError("Backbone image mean/std must contain three RGB values")
        if any(value <= 0 for value in self.backbone_image_std):
            raise ValueError("Backbone image std values must be positive")
