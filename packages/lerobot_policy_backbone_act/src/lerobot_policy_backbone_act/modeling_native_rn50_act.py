from __future__ import annotations

from lerobot.policies.act.modeling_act import ACTPolicy

from .configuration_native_rn50_act import ACTRN50FullConfig
from .modeling_backbone_act import _BackboneSpatialEncoder, _load_upstream_backbone


class ACTRN50FullPolicy(ACTPolicy):
    """Full-resolution dual-camera ACT centered on ours RN50."""

    config_class = ACTRN50FullConfig
    name = "act_rn50_full"

    def __init__(self, config: ACTRN50FullConfig, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        self.model.backbone = _BackboneSpatialEncoder(
            _load_upstream_backbone(config), config
        )


NativeRN50ACTPolicy = ACTRN50FullPolicy
