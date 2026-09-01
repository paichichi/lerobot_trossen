from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from torch import nn
from torch.nn import functional

from .configuration_backbone_act import BackboneACTConfig


def _checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    raise TypeError("Checkpoint args must be a mapping or namespace")


def _load_vit(config: BackboneACTConfig) -> nn.Module:
    checkpoint_path = Path(
        os.environ.get("BACKBONE_CHECKPOINT", config.backbone_checkpoint)
    ).expanduser().resolve()
    source_root = Path(
        os.environ.get("BACKBONE_SOURCE_ROOT", config.backbone_source_root)
    ).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")
    if not (source_root / "xirl/models.py").is_file():
        raise FileNotFoundError(f"TCC backbone source not found: {source_root}")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    models = importlib.import_module("xirl.models")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if config.backbone_family == "pretrained_vit":
        backbone = models.build_backbone(
            backbone="vit_b16",
            pretrain_path=str(checkpoint_path),
            train_norm_affine=False,
            train_adapters=False,
        )
    else:
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise TypeError("Ours ViT requires a TCC training checkpoint")
        backbone_name = str(_checkpoint_args(checkpoint).get("backbone", ""))
        if backbone_name not in {"vit", "vit_b16"}:
            raise ValueError(f"Expected a ViT-B/16 checkpoint, got {backbone_name!r}")
        backbone = models.build_backbone(
            backbone=backbone_name,
            pretrain_path="",
            train_norm_affine=False,
            train_adapters=False,
        )
        state = {
            key.removeprefix("backbone."): value
            for key, value in checkpoint["model"].items()
            if key.startswith("backbone.")
        }
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Backbone mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
            )

    if int(getattr(backbone, "output_dim", -1)) != 768:
        raise ValueError("The ViT backbone must expose 768-dimensional patch tokens")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    return backbone


class _ViTPatchEncoder(nn.Module):
    """Convert RGB images to the ViT 14x14 patch map consumed by ACT."""

    def __init__(self, backbone: nn.Module, config: BackboneACTConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = config.backbone_image_size
        self.register_buffer(
            "image_mean", torch.tensor(config.backbone_image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor(config.backbone_image_std).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True) -> _ViTPatchEncoder:
        super().train(mode)
        self.backbone.eval()
        return self

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW RGB images, got {tuple(images.shape)}")
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            images = functional.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (images - self.image_mean) / self.image_std

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        model = self.backbone.model
        patch_grid = model.conv_proj(self._preprocess(images))
        batch, channels, height, width = patch_grid.shape
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)
        class_token = model.class_token.expand(batch, -1, -1)
        encoded = model.encoder(torch.cat([class_token, patch_tokens], dim=1))[:, 1:]
        feature_map = encoded.transpose(1, 2).reshape(batch, channels, height, width)
        return {"feature_map": feature_map.contiguous()}


class BackboneACTPolicy(ACTPolicy):
    """Official LeRobot ACT with its image encoder replaced by a frozen ViT."""

    config_class = BackboneACTConfig
    name = "backbone_act"

    def __init__(self, config: BackboneACTConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.model.backbone = _ViTPatchEncoder(_load_vit(config), config)
        self.model.encoder_img_feat_input_proj = nn.Conv2d(768, config.dim_model, 1)
