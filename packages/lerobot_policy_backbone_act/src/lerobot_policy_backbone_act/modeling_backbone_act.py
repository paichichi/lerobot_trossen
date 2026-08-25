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
    raise TypeError("Upstream backbone checkpoint args must be a mapping or namespace")


def _load_upstream_backbone(config: BackboneACTConfig) -> nn.Module:
    checkpoint_path = Path(
        os.environ.get("BACKBONE_CHECKPOINT", config.backbone_checkpoint)
    ).expanduser().resolve()
    source_root = Path(
        os.environ.get("BACKBONE_SOURCE_ROOT", config.backbone_source_root)
    ).expanduser().resolve()
    models_path = source_root / "xirl" / "models.py"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Upstream backbone checkpoint not found: {checkpoint_path}"
        )
    if not models_path.is_file():
        raise FileNotFoundError(f"Backbone source checkout not found: {models_path}")

    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    models = importlib.import_module("xirl.models")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise TypeError("Expected an upstream training checkpoint with model weights")

    args = _checkpoint_args(checkpoint)
    backbone_name = str(args.get("backbone", ""))
    if "resnet50" not in backbone_name and "r3m" not in backbone_name:
        raise ValueError(
            "backbone_act ours_rn50 requires an RN50-family checkpoint; "
            f"checkpoint declares {backbone_name!r}"
        )

    backbone = models.build_backbone(
        backbone=backbone_name,
        pretrain_path="",
        train_norm_affine=False,
        train_adapters=False,
    )
    prefix = "backbone."
    state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise RuntimeError("Checkpoint contains no model.backbone weights")
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Upstream backbone checkpoint mismatch: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    if int(getattr(backbone, "output_dim", -1)) != 2048:
        raise ValueError("backbone_act ours_rn50 must expose 2048 channels")
    for parameter in backbone.parameters():
        parameter.requires_grad_(not config.freeze_vision_backbone)
    return backbone


class _BackboneSpatialEncoder(nn.Module):
    """Apply the upstream image contract and expose an RN50 layer4 map."""

    def __init__(self, backbone: nn.Module, config: BackboneACTConfig) -> None:
        super().__init__()
        self.backbone = backbone
        native_height = getattr(config, "backbone_image_height", None)
        native_width = getattr(config, "backbone_image_width", None)
        if (native_height is None) != (native_width is None):
            raise ValueError("Backbone image height and width must be configured together")
        self.image_shape = (
            (int(native_height), int(native_width))
            if native_height is not None
            else (config.backbone_image_size, config.backbone_image_size)
        )
        self.freeze_backbone = config.freeze_vision_backbone
        self.register_buffer(
            "image_mean",
            torch.tensor(config.backbone_image_mean).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(config.backbone_image_std).view(1, 3, 1, 1),
            persistent=True,
        )
        if self.freeze_backbone:
            self.backbone.eval()

    def train(self, mode: bool = True) -> _BackboneSpatialEncoder:
        super().train(mode)
        if self.freeze_backbone:
            # Freeze buffers and stochastic layers as well as parameters.
            self.backbone.eval()
        return self

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW RGB images, got {tuple(images.shape)}")
        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32).div_(255.0)
        elif torch.is_floating_point(images):
            images = images.to(dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported image dtype: {images.dtype}")
        if tuple(images.shape[-2:]) != self.image_shape:
            images = functional.interpolate(
                images,
                size=self.image_shape,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        mean = self.image_mean.to(dtype=images.dtype)
        std = self.image_std.to(dtype=images.dtype)
        return (images - mean) / std

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = self.preprocess(images)
        if hasattr(self.backbone, "forward_adapted"):
            feature_map = self.backbone.forward_adapted(images)
        elif hasattr(self.backbone, "forward_base") and hasattr(
            self.backbone, "apply_adapters"
        ):
            feature_map = self.backbone.apply_adapters(
                self.backbone.forward_base(images)
            )
        else:
            raise TypeError("ours_rn50 does not expose a spatial layer4 feature map")
        if feature_map.ndim != 4 or feature_map.shape[1] != 2048:
            raise RuntimeError(
                f"Expected Bx2048xHxW features, got {tuple(feature_map.shape)}"
            )
        return {"feature_map": feature_map}


class BackboneACTPolicy(ACTPolicy):
    """Official ACT with only its visual feature extractor replaced."""

    config_class = BackboneACTConfig
    name = "backbone_act"

    def __init__(self, config: BackboneACTConfig, **kwargs: Any) -> None:
        # Official ACT constructs its CVAE, Transformer, action objective, queue,
        # and a ResNet50-shaped image projection before this replacement.
        super().__init__(config, **kwargs)
        self.model.backbone = _BackboneSpatialEncoder(
            _load_upstream_backbone(config), config
        )
