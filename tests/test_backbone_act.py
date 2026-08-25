import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot_policy_backbone_act.configuration_backbone_act import BackboneACTConfig
from lerobot_policy_backbone_act.configuration_native_rn50_act import (
    NativeRN50ACTConfig,
)
from lerobot_policy_backbone_act.modeling_backbone_act import (
    BackboneACTPolicy,
    _BackboneSpatialEncoder,
    _load_upstream_backbone,
)
from torch import nn

from scripts.make_act_episode_order import shuffled_episode_order
from scripts.select_best_act_checkpoint import find_best_checkpoint


class _BackboneWithBatchNorm(nn.Module):
    output_dim = 2048

    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(3)


def _encoder_config(*, frozen: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        backbone_image_size=224,
        backbone_image_mean=(0.485, 0.456, 0.406),
        backbone_image_std=(0.229, 0.224, 0.225),
        freeze_vision_backbone=frozen,
    )


def _native_encoder_config() -> SimpleNamespace:
    return SimpleNamespace(
        backbone_image_height=480,
        backbone_image_width=640,
        backbone_image_mean=(0.485, 0.456, 0.406),
        backbone_image_std=(0.229, 0.224, 0.225),
        freeze_vision_backbone=False,
    )


def test_backbone_act_owns_visual_normalization() -> None:
    config = BackboneACTConfig()

    assert config.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY
    assert config.normalization_mapping["STATE"] == NormalizationMode.MEAN_STD
    assert config.normalization_mapping["ACTION"] == NormalizationMode.MEAN_STD


def test_backbone_preprocess_matches_uint8_and_unit_float_inputs() -> None:
    encoder = _BackboneSpatialEncoder(_BackboneWithBatchNorm(), _encoder_config())
    uint8_images = torch.randint(0, 256, (2, 3, 32, 48), dtype=torch.uint8)

    from_uint8 = encoder.preprocess(uint8_images)
    from_float = encoder.preprocess(uint8_images.float() / 255.0)

    assert from_uint8.shape == (2, 3, 224, 224)
    torch.testing.assert_close(from_uint8, from_float)


def test_native_rn50_preprocess_preserves_480x640_geometry() -> None:
    encoder = _BackboneSpatialEncoder(
        _BackboneWithBatchNorm(), _native_encoder_config()
    )
    images = torch.randint(0, 256, (2, 3, 480, 640), dtype=torch.uint8)

    processed = encoder.preprocess(images)

    assert processed.shape == (2, 3, 480, 640)


def test_native_rn50_is_locked_to_full_official_act() -> None:
    config = NativeRN50ACTConfig()

    assert config.backbone_image_height == 480
    assert config.backbone_image_width == 640
    assert not config.freeze_vision_backbone
    assert config.dim_model == 512
    assert config.n_heads == 8
    assert config.dim_feedforward == 3200
    assert config.n_encoder_layers == 4
    assert config.n_decoder_layers == 1
    assert config.n_vae_encoder_layers == 4
    assert config.normalization_mapping["VISUAL"] == NormalizationMode.IDENTITY


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("backbone_image_height", 224),
        ("backbone_image_width", 224),
        ("freeze_vision_backbone", True),
        ("dim_model", 256),
        ("n_heads", 4),
        ("dim_feedforward", 1024),
        ("n_encoder_layers", 2),
        ("n_vae_encoder_layers", 2),
    ],
)
def test_native_rn50_rejects_lite_or_resized_configuration(
    override: str, value: object
) -> None:
    with pytest.raises(ValueError, match="locked to native 480x640"):
        NativeRN50ACTConfig(**{override: value})


def test_frozen_backbone_stays_in_eval_mode_when_policy_trains() -> None:
    backbone = _BackboneWithBatchNorm()
    encoder = _BackboneSpatialEncoder(backbone, _encoder_config(frozen=True))

    encoder.train()

    assert encoder.training
    assert not backbone.training
    assert not backbone.batch_norm.training


def test_lite_release_rejects_vit_until_spatial_adapter_exists() -> None:
    with pytest.raises(ValueError, match="ViT requires"):
        BackboneACTConfig(backbone_family="ours_vit")


def test_deployment_environment_overrides_serialized_training_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    source_root = tmp_path / "TCC-core"
    checkpoint.touch()
    source_root.mkdir()
    monkeypatch.setenv("BACKBONE_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("BACKBONE_SOURCE_ROOT", str(source_root))

    with pytest.raises(FileNotFoundError, match=str(source_root / "xirl" / "models.py")):
        _load_upstream_backbone(
            BackboneACTConfig(
                backbone_checkpoint="/training/machine/missing.pt",
                backbone_source_root="/training/machine/TCC-core",
            )
        )


def test_act_split_is_seeded_random_and_stratified_per_task() -> None:
    tasks = [["carrot"]] * 100 + [["apple"]] * 100

    order, counts = shuffled_episode_order(tasks, seed=1000)

    assert counts == {"carrot": (80, 20), "apple": (80, 20)}
    assert order == shuffled_episode_order(tasks, seed=1000)[0]
    assert order[:100] != list(range(100))
    assert order[100:] != list(range(100, 200))


def test_best_act_checkpoint_uses_lowest_saved_validation_loss(tmp_path) -> None:
    output_dir = tmp_path / "run"
    (output_dir / "checkpoints" / "002000" / "pretrained_model").mkdir(parents=True)
    (output_dir / "checkpoints" / "004000" / "pretrained_model").mkdir(parents=True)
    (output_dir / "train.log").write_text(
        "step 2000: eval_loss=0.1250\nstep 4000: eval_loss=0.0910\n"
    )

    step, loss, checkpoint = find_best_checkpoint(output_dir)

    assert step == 4000
    assert loss == pytest.approx(0.091)
    assert checkpoint.name == "pretrained_model"


def test_trossen_contract_uses_official_scalar_cap_and_point_one_seconds() -> None:
    config_module = pytest.importorskip(
        "lerobot_robot_trossen.config_widowxai_follower"
    )
    config_class = config_module.WidowXAIFollowerConfig
    config = config_class(
        ip_address="192.168.1.4",
        loop_rate=20,
        min_time_to_move_multiplier=2.0,
        max_relative_target=0.07,
    )

    assert config.min_time_to_move_multiplier / config.loop_rate == pytest.approx(0.1)
    assert config.max_relative_target == pytest.approx(0.07)


@pytest.mark.skipif(
    not os.environ.get("ACT_LITE_REAL_BACKBONE_CHECKPOINT"),
    reason="real upstream checkpoint path not configured",
)
def test_real_ours_rn50_dual_camera_forward_backward_and_freeze() -> None:
    checkpoint = Path(os.environ["ACT_LITE_REAL_BACKBONE_CHECKPOINT"])
    source_root = Path(os.environ["ACT_LITE_REAL_BACKBONE_SOURCE_ROOT"])
    config = BackboneACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.images.cam_main": PolicyFeature(
                FeatureType.VISUAL, (3, 480, 640)
            ),
            "observation.images.cam_wrist": PolicyFeature(
                FeatureType.VISUAL, (3, 480, 640)
            ),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        device="cuda",
        push_to_hub=False,
        backbone_checkpoint=str(checkpoint),
        backbone_source_root=str(source_root),
        freeze_vision_backbone=True,
        chunk_size=40,
        n_action_steps=10,
        dim_model=256,
        n_heads=4,
        dim_feedforward=1024,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
    )
    policy = BackboneACTPolicy(config).cuda().train()
    batch_size = 2
    batch = {
        "observation.state": torch.randn(batch_size, 7, device="cuda"),
        "observation.images.cam_main": torch.randint(
            0, 256, (batch_size, 3, 480, 640), dtype=torch.uint8, device="cuda"
        ),
        "observation.images.cam_wrist": torch.randint(
            0, 256, (batch_size, 3, 480, 640), dtype=torch.uint8, device="cuda"
        ),
        "action": torch.randn(batch_size, 40, 7, device="cuda"),
        "action_is_pad": torch.zeros(batch_size, 40, dtype=torch.bool, device="cuda"),
    }

    loss, loss_dict = policy(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(loss_dict) == {"l1_loss", "kld_loss"}
    assert not policy.model.backbone.backbone.training
    assert not any(
        parameter.requires_grad or parameter.grad is not None
        for parameter in policy.model.backbone.backbone.parameters()
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in policy.named_parameters()
        if not name.startswith("model.backbone")
    )
