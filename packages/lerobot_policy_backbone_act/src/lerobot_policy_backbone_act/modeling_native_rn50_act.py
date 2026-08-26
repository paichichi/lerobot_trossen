from __future__ import annotations

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from torch import nn
from torch.nn import functional

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


def _automatic_carrot_heatmap_targets(
    images: torch.Tensor, output_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create soft orange-carrot heatmaps directly from RGB observations.

    The target is deliberately confidence gated. Ambiguous, tiny, or oversized
    masks receive no auxiliary loss instead of injecting a hard pseudo-label.
    This function is dataset-free and is used only while training/auditing.
    """
    if images.dtype == torch.uint8:
        rgb = images.float() / 255.0
    else:
        rgb = images.float().clamp(0.0, 1.0)
    red, green, blue = rgb.unbind(dim=1)
    value, max_index = rgb.max(dim=1)
    minimum = rgb.min(dim=1).values
    delta = value - minimum
    safe_delta = delta.clamp_min(1e-6)

    hue_r = torch.remainder((green - blue) / safe_delta, 6.0)
    hue_g = (blue - red) / safe_delta + 2.0
    hue_b = (red - green) / safe_delta + 4.0
    hue = (
        torch.where(max_index == 0, hue_r, torch.where(max_index == 1, hue_g, hue_b))
        / 6.0
    )
    saturation = delta / value.clamp_min(1e-6)

    orange = (
        (
            (hue >= 7.0 / 180.0)
            & (hue <= 13.0 / 180.0)
            & (saturation >= 140.0 / 255.0)
            & (value >= 150.0 / 255.0)
            & (red >= green + 0.04)
            & (green >= blue + 0.02)
        )
        .float()
        .unsqueeze(1)
    )
    orange = (functional.avg_pool2d(orange, 5, stride=1, padding=2) >= 0.2).float()
    pixel_mass = orange.sum(dim=(1, 2, 3))
    pooled = functional.adaptive_avg_pool2d(orange, output_size)
    valid = (
        (pixel_mass >= 200)
        & (pixel_mass <= 6000)
        & (pooled.amax(dim=(1, 2, 3)) >= 0.02)
    )
    target = pooled.flatten(1)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return target.reshape(-1, 1, *output_size), valid


class _SpatialGroundedVisualTokenAdapter(_ScaleCompatibleVisualTokenAdapter):
    """Scale-compatible adapter with an automatically supervised spatial gate."""

    def __init__(
        self,
        in_channels: int,
        dim_model: int,
        *,
        rms_eps: float,
        gain_init: float,
        attention_gain: float,
    ) -> None:
        super().__init__(in_channels, dim_model, rms_eps=rms_eps, gain_init=gain_init)
        self.heatmap_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.zeros_(self.heatmap_head.weight)
        nn.init.zeros_(self.heatmap_head.bias)
        self.attention_gain = attention_gain
        self._call_index = 0
        self._main_camera_index = 0
        self.last_main_heatmap_logits: torch.Tensor | None = None

    def reset_forward_cache(self, main_camera_index: int) -> None:
        self._call_index = 0
        self._main_camera_index = main_camera_index
        self.last_main_heatmap_logits = None

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        call_index = self._call_index
        self._call_index += 1
        rms = (
            feature_map.float()
            .square()
            .mean(dim=1, keepdim=True)
            .add(self.rms_eps)
            .sqrt()
        )
        normalized = (feature_map.float() / rms).to(dtype=feature_map.dtype)
        tokens = self.projection(normalized)
        tokens = self.token_norm(tokens.permute(0, 2, 3, 1)) * self.token_gain
        tokens = tokens.permute(0, 3, 1, 2).contiguous()
        if call_index == self._main_camera_index:
            logits = self.heatmap_head(normalized)
            self.last_main_heatmap_logits = logits
            centered_attention = 2.0 * logits.sigmoid() - 1.0
            tokens = tokens * (1.0 + self.attention_gain * centered_attention)
        return tokens


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
            self.model.encoder_img_feat_input_proj = _ScaleCompatibleVisualTokenAdapter(
                in_channels=2048,
                dim_model=config.dim_model,
                rms_eps=config.visual_adapter_rms_eps,
                gain_init=config.visual_token_gain_init,
            )
        elif config.visual_adapter_version == "spatial_grounded_v1":
            self.model.encoder_img_feat_input_proj = _SpatialGroundedVisualTokenAdapter(
                in_channels=2048,
                dim_model=config.dim_model,
                rms_eps=config.visual_adapter_rms_eps,
                gain_init=config.visual_token_gain_init,
                attention_gain=config.spatial_attention_gain,
            )

    def _reset_spatial_adapter(
        self,
    ) -> tuple[_SpatialGroundedVisualTokenAdapter | None, int]:
        adapter = self.model.encoder_img_feat_input_proj
        if not isinstance(adapter, _SpatialGroundedVisualTokenAdapter):
            return None, -1
        camera_keys = list(self.config.image_features)
        main_indices = [
            index for index, key in enumerate(camera_keys) if key.endswith("cam_main")
        ]
        if len(main_indices) != 1:
            raise RuntimeError(
                f"Expected exactly one cam_main input, found {camera_keys}"
            )
        adapter.reset_forward_cache(main_indices[0])
        return adapter, main_indices[0]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self._reset_spatial_adapter()
        return super().predict_action_chunk(batch)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        adapter, main_index = self._reset_spatial_adapter()
        loss, loss_dict = super().forward(batch)
        if adapter is None or adapter.last_main_heatmap_logits is None:
            return loss, loss_dict

        main_key = list(self.config.image_features)[main_index]
        logits = adapter.last_main_heatmap_logits
        targets, valid = _automatic_carrot_heatmap_targets(
            batch[main_key], tuple(logits.shape[-2:])
        )
        if "frame_index" in batch:
            valid = valid & (
                batch["frame_index"] < self.config.spatial_supervision_max_frame
            )
        per_sample = -(
            targets.flatten(1)
            * functional.log_softmax(logits.float().flatten(1), dim=1)
        ).sum(dim=1)
        valid_float = valid.to(dtype=per_sample.dtype)
        heatmap_loss = (per_sample * valid_float).sum() / valid_float.sum().clamp_min(
            1.0
        )
        loss = loss + self.config.spatial_heatmap_loss_weight * heatmap_loss
        loss_dict["spatial_heatmap_loss"] = heatmap_loss.item()
        loss_dict["spatial_heatmap_valid_fraction"] = valid_float.mean().item()
        return loss, loss_dict


NativeRN50ACTPolicy = ACTRN50FullPolicy
