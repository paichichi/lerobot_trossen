from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from lerobot.policies.act.modeling_act import ACTPolicy

from .configuration_tcc_act import TCCACTConfig


def _checkpoint_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    args = checkpoint.get("args", {})
    if isinstance(args, dict):
        return args
    if hasattr(args, "__dict__"):
        return vars(args)
    raise TypeError("TCC checkpoint args must be a mapping or argparse namespace")


def _load_tcc_backbone(config: TCCACTConfig) -> nn.Module:
    checkpoint_path = Path(config.tcc_backbone_checkpoint).expanduser().resolve()
    source_root = Path(config.tcc_source_root).expanduser().resolve()
    models_path = source_root / "xirl" / "models.py"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TCC backbone checkpoint not found: {checkpoint_path}")
    if not models_path.is_file():
        raise FileNotFoundError(f"TCC source checkout not found: {models_path}")

    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    models = importlib.import_module("xirl.models")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise TypeError("Expected a TCC training checkpoint containing model weights")

    args = _checkpoint_args(checkpoint)
    backbone_name = str(args.get("backbone", ""))
    if "resnet50" not in backbone_name and "r3m" not in backbone_name:
        raise ValueError(
            f"TCCACT requires the ours RN50 family, checkpoint declares {backbone_name!r}"
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
            "TCC backbone checkpoint mismatch: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )

    if int(getattr(backbone, "output_dim", -1)) != 2048:
        raise ValueError("TCCACT ours_rn50 backbone must expose 2048 channels")
    for parameter in backbone.parameters():
        parameter.requires_grad_(not config.freeze_vision_backbone)
    return backbone


class _TCCSpatialBackbone(nn.Module):
    """Adapt the TCC RN50 layer4 map to upstream ACT's backbone contract."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if hasattr(self.backbone, "forward_adapted"):
            feature_map = self.backbone.forward_adapted(images)
        elif hasattr(self.backbone, "forward_base") and hasattr(
            self.backbone, "apply_adapters"
        ):
            feature_map = self.backbone.apply_adapters(
                self.backbone.forward_base(images)
            )
        else:
            raise TypeError(
                "TCC RN50 backbone does not expose a spatial layer4 feature map"
            )
        if feature_map.ndim != 4 or feature_map.shape[1] != 2048:
            raise RuntimeError(
                f"Expected Bx2048xHxW TCC features, got {tuple(feature_map.shape)}"
            )
        return {"feature_map": feature_map}


class TCCACTPolicy(ACTPolicy):
    """Upstream ACT policy with only its visual feature extractor replaced."""

    config_class = TCCACTConfig
    name = "tcc_act"

    def __init__(self, config: TCCACTConfig, **kwargs: Any) -> None:
        # This constructs the official ACT model, including its ResNet50-shaped
        # image projection. Replacing only model.backbone preserves every ACT
        # component and training objective downstream of the spatial tokens.
        super().__init__(config, **kwargs)
        self.model.backbone = _TCCSpatialBackbone(_load_tcc_backbone(config))
