from __future__ import annotations

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from torch import nn

from .configuration_native_rn50_act import ACTRN50FullConfig
from .modeling_backbone_act import _BackboneSpatialEncoder, _load_upstream_backbone


class _ScaleCompatibleVisualTokenAdapter(nn.Module):
    """Convert small-magnitude RN50 maps into well-scaled ACT visual tokens.

    The upstream adapted RN50 emits spatial features whose absolute scale is
    much smaller than the official RN18 features. Normalize each spatial
    feature vector before projection, then normalize each projected token so
    visual content reaches ACT at a scale comparable to its state token.
    """

    def __init__(
        self,
        in_channels: int,
        dim_model: int,
        *,
        rms_eps: float,
        gain_init: float,
    ) -> None:
        super().__init__()
        self.rms_eps = rms_eps
        self.projection = nn.Conv2d(in_channels, dim_model, kernel_size=1)
        self.token_norm = nn.LayerNorm(dim_model)
        self.token_gain = nn.Parameter(torch.tensor(float(gain_init)))

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        # Accumulate the norm in fp32 so AMP cannot underflow on the adapted
        # RN50 features, then return to the model dtype for the projection.
        rms = (
            feature_map.float()
            .square()
            .mean(dim=1, keepdim=True)
            .add(self.rms_eps)
            .sqrt()
        )
        normalized = (feature_map.float() / rms).to(dtype=feature_map.dtype)
        tokens = self.projection(normalized)
        tokens = tokens.permute(0, 2, 3, 1)
        tokens = self.token_norm(tokens) * self.token_gain
        return tokens.permute(0, 3, 1, 2).contiguous()


class ACTRN50FullPolicy(ACTPolicy):
    """Full-resolution dual-camera ACT centered on ours RN50."""

    config_class = ACTRN50FullConfig
    name = "act_rn50_full"

    def __init__(self, config: ACTRN50FullConfig, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        self.model.backbone = _BackboneSpatialEncoder(
            _load_upstream_backbone(config), config
        )
        if config.visual_adapter_version == "rms_ln_v1":
            self.model.encoder_img_feat_input_proj = (
                _ScaleCompatibleVisualTokenAdapter(
                    in_channels=2048,
                    dim_model=config.dim_model,
                    rms_eps=config.visual_adapter_rms_eps,
                    gain_init=config.visual_token_gain_init,
                )
            )


NativeRN50ACTPolicy = ACTRN50FullPolicy
