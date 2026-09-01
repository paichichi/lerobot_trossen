from types import SimpleNamespace

import pytest
import torch
from lerobot.configs import NormalizationMode
from lerobot_policy_backbone_act.configuration_backbone_act import BackboneACTConfig
from lerobot_policy_backbone_act.modeling_backbone_act import _ViTPatchEncoder
from torch import nn


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Identity()


def test_policy_supports_only_frozen_vit_backbones() -> None:
    for family in ("ours_vit", "pretrained_vit"):
        config = BackboneACTConfig(backbone_family=family)
        assert config.freeze_vision_backbone
        assert config.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY

    with pytest.raises(ValueError, match="backbone_family"):
        BackboneACTConfig(backbone_family="unsupported")
    with pytest.raises(ValueError, match="frozen"):
        BackboneACTConfig(backbone_family="ours_vit", freeze_vision_backbone=False)


def test_vit_preprocessing_matches_uint8_and_unit_float() -> None:
    config = SimpleNamespace(
        backbone_image_size=224,
        backbone_image_mean=(0.485, 0.456, 0.406),
        backbone_image_std=(0.229, 0.224, 0.225),
    )
    encoder = _ViTPatchEncoder(FakeBackbone(), config)
    image = torch.randint(0, 256, (2, 3, 64, 80), dtype=torch.uint8)
    torch.testing.assert_close(
        encoder._preprocess(image),
        encoder._preprocess(image.float() / 255.0),
    )
    assert encoder._preprocess(image).shape == (2, 3, 224, 224)
