from __future__ import annotations

from lerobot.policies.act.modeling_act import ACTPolicy

from .configuration_native_rn50_act import NativeRN50ACTConfig
from .modeling_backbone_act import _BackboneSpatialEncoder, _load_upstream_backbone


class NativeRN50ACTPolicy(ACTPolicy):
    """Full official ACT with only ResNet18 replaced by ours RN50."""

    config_class = NativeRN50ACTConfig
    name = "native_rn50_act"

    def __init__(self, config: NativeRN50ACTConfig, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        self.model.backbone = _BackboneSpatialEncoder(
            _load_upstream_backbone(config), config
        )
