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


def _configure_backbone_trainability(
    backbone: nn.Module, config: BackboneACTConfig
) -> None:
    """Apply one explicit backbone trainability contract without ambiguity."""
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    if config.freeze_vision_backbone:
        return

    if config.vit_train_layer_norm_only:
        layer_norm_count = 0
        for module in backbone.modules():
            if isinstance(module, nn.LayerNorm):
                layer_norm_count += 1
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(True)
        if layer_norm_count == 0:
            raise TypeError("LayerNorm-only ViT tuning found no LayerNorm modules")
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in backbone.parameters()
            if parameter.requires_grad
        )
        if (
            getattr(backbone, "output_dim", None) == 768
            and trainable_parameter_count != 38_400
        ):
            raise RuntimeError(
                "ViT-B/16 LayerNorm-only tuning must expose exactly 38,400 "
                f"backbone parameters, got {trainable_parameter_count:,}"
            )
        return

    trainable_last_blocks = config.vit_trainable_last_blocks
    if trainable_last_blocks is None:
        for parameter in backbone.parameters():
            parameter.requires_grad_(True)
        return

    model = getattr(backbone, "model", None)
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    final_layer_norm = getattr(encoder, "ln", None)
    if layers is None or final_layer_norm is None:
        raise TypeError(
            "Selective ours_vit tuning requires encoder.layers and encoder.ln"
        )
    layer_list = list(layers)
    if trainable_last_blocks > len(layer_list):
        raise ValueError(
            f"Requested {trainable_last_blocks} trainable ViT blocks, "
            f"but the backbone only has {len(layer_list)}"
        )
    for layer in layer_list[-trainable_last_blocks:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for parameter in final_layer_norm.parameters():
        parameter.requires_grad_(True)


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

    if config.backbone_family == "pretrained_vit":
        backbone = models.build_backbone(
            backbone="vit_b16",
            pretrain_path=str(checkpoint_path),
            train_norm_affine=False,
            train_adapters=False,
        )
        source_state = models._unwrap_state_dict(checkpoint)
        if not isinstance(source_state, dict):
            raise TypeError("Expected raw pretrained ViT weights to be a mapping")
        target_state = backbone.model.state_dict()
        loaded_target_keys: set[str] = set()
        for key, value in source_state.items():
            if not isinstance(value, torch.Tensor):
                continue
            candidates = (
                key,
                key.removeprefix("module."),
                key.removeprefix("backbone."),
                key.removeprefix("backbone.model."),
            )
            mapped_key = next(
                (candidate for candidate in candidates if candidate in target_state),
                None,
            )
            if mapped_key is None:
                mapped_key = models._mae_to_torchvision_vit_key(key)
            if (
                mapped_key in target_state
                and target_state[mapped_key].shape == value.shape
            ):
                loaded_target_keys.add(mapped_key)
        missing_target_keys = sorted(set(target_state) - loaded_target_keys)
        if missing_target_keys:
            raise RuntimeError(
                "Raw pretrained ViT checkpoint did not fully initialize the "
                f"encoder; missing={missing_target_keys[:20]}"
            )
        if len(loaded_target_keys) != 150:
            raise RuntimeError(
                "ViT-B/16 raw checkpoint must initialize exactly 150 encoder "
                f"tensors, got {len(loaded_target_keys)}"
            )
        _configure_backbone_trainability(backbone, config)
        return backbone

    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise TypeError("Expected an upstream training checkpoint with model weights")

    args = _checkpoint_args(checkpoint)
    backbone_name = str(args.get("backbone", ""))
    is_rn50 = "resnet50" in backbone_name or "r3m" in backbone_name
    is_vit = backbone_name in {"vit", "vit_b16"}
    if config.backbone_family == "ours_rn50" and not is_rn50:
        raise ValueError(
            "backbone_act ours_rn50 requires an RN50-family checkpoint; "
            f"checkpoint declares {backbone_name!r}"
        )
    if config.backbone_family == "ours_vit" and not is_vit:
        raise ValueError(
            "backbone_act ours_vit requires a ViT-B/16 checkpoint; "
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

    expected_output_dim = 768 if config.backbone_family == "ours_vit" else 2048
    if int(getattr(backbone, "output_dim", -1)) != expected_output_dim:
        raise ValueError(
            f"backbone_act {config.backbone_family} must expose "
            f"{expected_output_dim} channels"
        )
    _configure_backbone_trainability(backbone, config)
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
        self.trainable_vit_last_blocks = getattr(
            config, "vit_trainable_last_blocks", None
        )
        self.train_vit_layer_norm_only = getattr(
            config, "vit_train_layer_norm_only", False
        )
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
        elif self.trainable_vit_last_blocks is not None:
            # Frozen early blocks must also keep deterministic inference
            # behavior. Only the selected final blocks enter training mode.
            self.backbone.eval()
            encoder = self.backbone.model.encoder
            for layer in list(encoder.layers)[-self.trainable_vit_last_blocks :]:
                layer.train(mode)
            encoder.ln.train(mode)
        elif self.train_vit_layer_norm_only:
            # LayerNorm has no running statistics. Keep the whole ViT in eval
            # mode so frozen dropout/stochastic layers cannot drift, then mark
            # only LayerNorm modules with the requested public mode.
            self.backbone.eval()
            for module in self.backbone.modules():
                if isinstance(module, nn.LayerNorm):
                    module.train(mode)
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


class _ViTPatchSpatialEncoder(_BackboneSpatialEncoder):
    """Expose the TCC ViT-B/16 patch tokens as a 14x14 ACT feature map."""

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = self.preprocess(images)
        model = getattr(self.backbone, "model", None)
        if model is None or not all(
            hasattr(model, attribute)
            for attribute in ("conv_proj", "class_token", "encoder")
        ):
            raise TypeError(
                "ours_vit must expose the torchvision ViT model as backbone.model"
            )

        patch_grid = model.conv_proj(images)
        batch_size, channels, grid_height, grid_width = patch_grid.shape
        if channels != 768:
            raise RuntimeError(
                f"Expected 768-channel ViT patch embeddings, got {channels}"
            )
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)
        class_token = model.class_token.expand(batch_size, -1, -1)
        encoded_tokens = model.encoder(torch.cat([class_token, patch_tokens], dim=1))
        patch_tokens = encoded_tokens[:, 1:]
        expected_tokens = grid_height * grid_width
        if patch_tokens.shape != (batch_size, expected_tokens, channels):
            raise RuntimeError(
                "Unexpected ViT patch token shape: "
                f"got {tuple(patch_tokens.shape)}, expected "
                f"{(batch_size, expected_tokens, channels)}"
            )
        feature_map = (
            patch_tokens.transpose(1, 2)
            .reshape(batch_size, channels, grid_height, grid_width)
            .contiguous()
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
        backbone = _load_upstream_backbone(config)
        if config.backbone_family in {"ours_vit", "pretrained_vit"}:
            self.model.backbone = _ViTPatchSpatialEncoder(backbone, config)
            self.model.encoder_img_feat_input_proj = nn.Conv2d(
                768, config.dim_model, kernel_size=1
            )
        else:
            self.model.backbone = _BackboneSpatialEncoder(backbone, config)
