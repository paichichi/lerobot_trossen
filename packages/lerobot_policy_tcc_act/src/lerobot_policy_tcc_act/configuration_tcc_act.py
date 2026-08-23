from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("tcc_act")
@dataclass
class TCCACTConfig(ACTConfig):
    """ACT whose spatial image tokens come from a trained TCC backbone.

    ACT's CVAE, Transformer, action-chunk objective, normalization, and runtime
    queue remain the upstream LeRobot implementation.
    """

    tcc_backbone_checkpoint: str = ""
    tcc_source_root: str = ""
    freeze_vision_backbone: bool = True

    # The projection allocated by upstream ACT must match an RN50 layer4 map.
    vision_backbone: str = "resnet50"
    pretrained_backbone_weights: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.vision_backbone != "resnet50":
            raise ValueError("TCCACT currently supports the ours_rn50 backbone only")
