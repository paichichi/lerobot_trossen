from __future__ import annotations

import einops
import torch
from lerobot.policies.act.modeling_act import ACT, ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

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


class _VisualGoalACT(ACT):
    """ACT whose control token must pass through a visual-only goal bottleneck.

    The two camera streams remain full spatial token maps. A learned query
    cross-attends to those maps before robot state is introduced. The resulting
    visual goal and projected state are fused into one control token; there is
    no independent state token in the policy encoder.
    """

    def __init__(self, config: ACTRN50FullConfig) -> None:
        super().__init__(config)
        self.visual_goal_query = nn.Parameter(torch.empty(1, 1, config.dim_model))
        self.visual_goal_attention = nn.MultiheadAttention(
            config.dim_model, config.n_heads, dropout=config.dropout
        )
        self.visual_goal_norm = nn.LayerNorm(config.dim_model)
        self.control_token_norm = nn.LayerNorm(config.dim_model)
        self.visual_goal_pos_embed = nn.Parameter(torch.empty(1, 1, config.dim_model))
        nn.init.normal_(self.visual_goal_query, std=0.02)
        nn.init.normal_(self.visual_goal_pos_embed, std=0.02)

    def _encode_visual_goal(
        self,
        visual_tokens: Tensor,
        visual_pos_embed: Tensor,
    ) -> Tensor:
        batch_size = visual_tokens.shape[1]
        query = self.visual_goal_query.expand(-1, batch_size, -1)
        attended = self.visual_goal_attention(
            query=query,
            key=visual_tokens + visual_pos_embed,
            value=visual_tokens,
            need_weights=False,
        )[0]
        return self.visual_goal_norm(query + attended)

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """Run the official ACT computation with one visual-first fusion step."""
        if self.config.use_vae and self.training:
            assert ACTION in batch

        batch_size = (
            batch[OBS_IMAGES][0].shape[0]
            if OBS_IMAGES in batch
            else batch[OBS_ENV_STATE].shape[0]
        )

        # This is the unmodified official ACT VAE path.
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(
                    batch[OBS_STATE]
                ).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
            pos_embed = self.vae_encoder_pos_enc.clone().detach()
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[ACTION].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(
                [batch_size, self.config.latent_dim],
                dtype=torch.float32,
                device=(
                    batch[OBS_STATE].device
                    if OBS_STATE in batch
                    else batch[OBS_IMAGES][0].device
                ),
            )

        visual_tokens: list[Tensor] = []
        visual_pos_embeddings: list[Tensor] = []
        for image in batch[OBS_IMAGES]:
            camera_features = self.backbone(image)["feature_map"]
            camera_pos = self.encoder_cam_feat_pos_embed(camera_features).to(
                dtype=camera_features.dtype
            )
            camera_features = self.encoder_img_feat_input_proj(camera_features)
            visual_tokens.append(
                einops.rearrange(camera_features, "b c h w -> (h w) b c")
            )
            visual_pos_embeddings.append(
                einops.rearrange(camera_pos, "b c h w -> (h w) b c")
            )
        all_visual_tokens = torch.cat(visual_tokens, dim=0)
        all_visual_pos = torch.cat(visual_pos_embeddings, dim=0)
        visual_goal = self._encode_visual_goal(all_visual_tokens, all_visual_pos)

        encoder_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_pos = [self.encoder_1d_feature_pos_embed.weight[0]]
        if self.config.robot_state_feature:
            state = self.encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(0)
            control_token = self.control_token_norm(visual_goal + state).squeeze(0)
            encoder_tokens.append(control_token)
            encoder_pos.append(self.visual_goal_pos_embed.squeeze(0).squeeze(0))
        else:
            encoder_tokens.append(visual_goal.squeeze(0))
            encoder_pos.append(self.visual_goal_pos_embed.squeeze(0).squeeze(0))
        if self.config.env_state_feature:
            encoder_tokens.append(
                self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            )
            # Preserve the official ordering for the optional environment token.
            encoder_pos.append(self.encoder_1d_feature_pos_embed.weight[-1])

        encoder_in_tokens = torch.cat(
            [torch.stack(encoder_tokens, dim=0), all_visual_tokens], dim=0
        )
        one_d_pos = torch.stack(encoder_pos, dim=0).unsqueeze(1)
        encoder_in_pos_embed = torch.cat([one_d_pos, all_visual_pos], dim=0)

        encoder_out = self.encoder(
            encoder_in_tokens, pos_embed=encoder_in_pos_embed
        )
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        ).transpose(0, 1)
        return self.action_head(decoder_out), (mu, log_sigma_x2)


class ACTRN50FullPolicy(ACTPolicy):
    """Full-resolution dual-camera ACT centered on ours RN50."""

    config_class = ACTRN50FullConfig
    name = "act_rn50_full"

    def __init__(self, config: ACTRN50FullConfig, **kwargs: object) -> None:
        super().__init__(config, **kwargs)
        if config.visual_goal_version == "visual_goal_v1":
            self.model = _VisualGoalACT(config)
        self.model.backbone = _BackboneSpatialEncoder(
            _load_upstream_backbone(config), config
        )
        if config.visual_adapter_version == "rms_ln_v1" or (
            config.visual_goal_version == "visual_goal_v1"
        ):
            self.model.encoder_img_feat_input_proj = (
                _ScaleCompatibleVisualTokenAdapter(
                    in_channels=2048,
                    dim_model=config.dim_model,
                    rms_eps=config.visual_adapter_rms_eps,
                    gain_init=config.visual_token_gain_init,
                )
            )


NativeRN50ACTPolicy = ACTRN50FullPolicy
