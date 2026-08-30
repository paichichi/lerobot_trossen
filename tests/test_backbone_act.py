import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot_policy_backbone_act.configuration_backbone_act import BackboneACTConfig
from lerobot_policy_backbone_act.configuration_native_rn50_act import (
    ACTRN50FullConfig,
)
from lerobot_policy_backbone_act.modeling_backbone_act import (
    BackboneACTPolicy,
    _BackboneSpatialEncoder,
    _configure_backbone_trainability,
    _load_upstream_backbone,
    _ViTPatchSpatialEncoder,
)
from lerobot_policy_backbone_act.modeling_native_rn50_act import (
    _ScaleCompatibleVisualTokenAdapter,
)
from torch import nn

from scripts.eval_act_image_swap import dispersion
from scripts.make_act_episode_order import shuffled_episode_order
from scripts.select_best_act_checkpoint import find_best_checkpoint


class _BackboneWithBatchNorm(nn.Module):
    output_dim = 2048

    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(3)


class _FakeViTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.class_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.encoder = nn.Identity()


class _FakeViTBackbone(nn.Module):
    output_dim = 768

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeViTModel()


class _FakeSelectiveViTEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(*(nn.Linear(2, 2) for _ in range(12)))
        self.ln = nn.LayerNorm(2)


class _FakeSelectiveViTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_proj = nn.Conv2d(3, 2, kernel_size=1)
        self.encoder = _FakeSelectiveViTEncoder()


class _FakeSelectiveViTBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeSelectiveViTModel()


def _encoder_config(*, frozen: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        backbone_image_size=224,
        backbone_image_mean=(0.485, 0.456, 0.406),
        backbone_image_std=(0.229, 0.224, 0.225),
        freeze_vision_backbone=frozen,
        vit_trainable_last_blocks=None,
        vit_train_layer_norm_only=False,
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
    config = ACTRN50FullConfig()

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
    assert config.visual_adapter_version == "legacy"
    assert config.visual_token_gain_init == 0.5


def test_scale_compatible_visual_adapter_normalizes_token_scale() -> None:
    torch.manual_seed(0)
    adapter = _ScaleCompatibleVisualTokenAdapter(
        2048, 512, rms_eps=1e-6, gain_init=1.0
    )
    feature_map = torch.randn(2, 2048, 3, 4) * 0.006

    tokens = adapter(feature_map)
    token_rms = tokens.float().square().mean(dim=1).sqrt()

    assert tokens.shape == (2, 512, 3, 4)
    torch.testing.assert_close(token_rms, torch.ones_like(token_rms), atol=2e-4, rtol=2e-4)
    assert torch.isfinite(tokens).all()


def test_scale_compatible_visual_adapter_is_stable_to_backbone_scale() -> None:
    torch.manual_seed(0)
    adapter = _ScaleCompatibleVisualTokenAdapter(
        2048, 512, rms_eps=1e-12, gain_init=1.0
    ).eval()
    feature_map = torch.randn(2, 2048, 3, 4)

    small = adapter(feature_map * 0.006)
    large = adapter(feature_map * 0.6)

    torch.testing.assert_close(small, large, atol=2e-5, rtol=2e-5)


def test_native_rn50_rejects_unknown_visual_adapter() -> None:
    with pytest.raises(ValueError, match="visual_adapter_version"):
        ACTRN50FullConfig(visual_adapter_version="unknown")


def test_full_adapter_v1_uses_official_act_projection_contract() -> None:
    config = ACTRN50FullConfig(visual_adapter_version="full_adapter_v1")

    assert config.visual_adapter_version == "full_adapter_v1"


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("backbone_image_height", 224),
        ("backbone_image_width", 224),
        ("freeze_vision_backbone", True),
    ],
)
def test_act_rn50_full_rejects_broken_visual_contract(
    override: str, value: object
) -> None:
    with pytest.raises(ValueError, match="native trainable ours RN50 contract"):
        ACTRN50FullConfig(**{override: value})


def test_act_rn50_full_capacity_is_tunable_without_changing_visual_contract() -> None:
    config = ACTRN50FullConfig(
        dim_model=768,
        n_heads=12,
        dim_feedforward=4096,
        n_encoder_layers=6,
        n_vae_encoder_layers=6,
    )

    assert config.dim_model == 768
    assert config.n_heads == 12
    assert config.dim_feedforward == 4096
    assert config.n_encoder_layers == 6
    assert config.n_vae_encoder_layers == 6


def test_act_rn50_full_augmentation_is_photometric_only() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "act_rn50_full_image_transforms.json"
    )
    transforms = json.loads(config_path.read_text())

    assert set(transforms) == {
        "brightness",
        "contrast",
        "saturation",
        "hue",
        "sharpness",
    }
    assert all(transform["type"] != "RandomAffine" for transform in transforms.values())


def test_image_swap_dispersion_detects_output_collapse() -> None:
    collapsed = np.zeros((4, 10, 7), dtype=np.float32)
    responsive = collapsed.copy()
    responsive[:, :, 0] = np.arange(4, dtype=np.float32)[:, None]

    assert dispersion(collapsed)["rms_about_mean"] == 0.0
    assert dispersion(responsive)["rms_about_mean"] > 0.0


def test_frozen_backbone_stays_in_eval_mode_when_policy_trains() -> None:
    backbone = _BackboneWithBatchNorm()
    encoder = _BackboneSpatialEncoder(backbone, _encoder_config(frozen=True))

    encoder.train()

    assert encoder.training
    assert not backbone.training
    assert not backbone.batch_norm.training


def test_backbone_act_accepts_fixed_resolution_ours_vit() -> None:
    config = BackboneACTConfig(backbone_family="ours_vit")

    assert config.backbone_image_size == 224
    with pytest.raises(ValueError, match="fixed 224x224"):
        BackboneACTConfig(backbone_family="ours_vit", backbone_image_size=384)
    pretrained = BackboneACTConfig(backbone_family="pretrained_vit")
    assert pretrained.backbone_image_size == 224
    with pytest.raises(ValueError, match="fixed 224x224"):
        BackboneACTConfig(backbone_family="pretrained_vit", backbone_image_size=384)


def test_ours_vit_selective_tuning_rejects_ambiguous_configs() -> None:
    with pytest.raises(ValueError, match="requires freeze_vision_backbone=false"):
        BackboneACTConfig(
            backbone_family="ours_vit",
            freeze_vision_backbone=True,
            vit_trainable_last_blocks=1,
        )
    with pytest.raises(ValueError, match="between 1 and 12"):
        BackboneACTConfig(
            backbone_family="ours_vit",
            freeze_vision_backbone=False,
            vit_trainable_last_blocks=0,
        )
    with pytest.raises(ValueError, match="only valid"):
        BackboneACTConfig(
            backbone_family="ours_rn50",
            freeze_vision_backbone=False,
            vit_trainable_last_blocks=1,
        )
    with pytest.raises(ValueError, match="requires freeze_vision_backbone=false"):
        BackboneACTConfig(
            backbone_family="ours_vit",
            freeze_vision_backbone=True,
            vit_train_layer_norm_only=True,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        BackboneACTConfig(
            backbone_family="ours_vit",
            freeze_vision_backbone=False,
            vit_trainable_last_blocks=1,
            vit_train_layer_norm_only=True,
        )


def test_ours_vit_selective_tuning_only_unfreezes_last_block_and_final_ln() -> None:
    backbone = _FakeSelectiveViTBackbone()
    config = BackboneACTConfig(
        backbone_family="ours_vit",
        freeze_vision_backbone=False,
        vit_trainable_last_blocks=1,
    )

    _configure_backbone_trainability(backbone, config)

    layers = list(backbone.model.encoder.layers)
    assert not any(parameter.requires_grad for layer in layers[:-1] for parameter in layer.parameters())
    assert all(parameter.requires_grad for parameter in layers[-1].parameters())
    assert all(parameter.requires_grad for parameter in backbone.model.encoder.ln.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.model.conv_proj.parameters())


def test_ours_vit_layer_norm_only_tuning_freezes_every_non_ln_parameter() -> None:
    backbone = _FakeSelectiveViTBackbone()
    config = BackboneACTConfig(
        backbone_family="ours_vit",
        freeze_vision_backbone=False,
        vit_train_layer_norm_only=True,
    )

    _configure_backbone_trainability(backbone, config)

    trainable_names = {
        name for name, parameter in backbone.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {
        "model.encoder.ln.weight",
        "model.encoder.ln.bias",
    }


def test_ours_vit_layer_norm_only_keeps_non_ln_modules_in_eval_mode() -> None:
    config = _encoder_config(frozen=False)
    config.vit_train_layer_norm_only = True
    backbone = _FakeSelectiveViTBackbone()
    encoder = _BackboneSpatialEncoder(backbone, config)

    encoder.train()

    assert encoder.training
    assert not backbone.training
    assert backbone.model.encoder.ln.training
    assert not backbone.model.conv_proj.training


def test_ours_vit_exposes_spatial_patch_tokens_without_cls() -> None:
    encoder = _ViTPatchSpatialEncoder(
        _FakeViTBackbone(), _encoder_config(frozen=False)
    )
    images = torch.randn(2, 3, 224, 224)

    feature_map = encoder(images)["feature_map"]

    assert feature_map.shape == (2, 768, 14, 14)
    expected = encoder.backbone.model.conv_proj(encoder.preprocess(images))
    torch.testing.assert_close(feature_map, expected)


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

    order, counts = shuffled_episode_order(tasks, seed=1000, eval_split=0.2)

    assert counts == {"carrot": (80, 20), "apple": (80, 20)}
    assert order == shuffled_episode_order(tasks, seed=1000, eval_split=0.2)[0]
    assert order[:100] != list(range(100))
    assert order[100:] != list(range(100, 200))


def test_act_split_can_train_on_every_episode() -> None:
    tasks = [["carrot"]] * 100

    order, counts = shuffled_episode_order(tasks, seed=1000, eval_split=0.0)

    assert sorted(order) == list(range(100))
    assert counts == {"carrot": (100, 0)}


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


def test_best_act_checkpoint_rejects_visually_collapsed_model(tmp_path) -> None:
    output_dir = tmp_path / "run"
    image_swap_dir = output_dir / "image_swap"
    image_swap_dir.mkdir(parents=True)
    (output_dir / "train.log").write_text(
        "step 2000: eval_loss=0.0800\nstep 4000: eval_loss=0.0900\n"
    )
    for step in (2000, 4000):
        (output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model").mkdir(
            parents=True
        )
    for step, paired, main in ((2000, 0.02, 0.01), (4000, 0.61, 0.57)):
        report = {
            "predicted_action_dispersion": {
                "paired_images": {"dispersion_ratio_vs_recorded": paired},
                "main_only": {"dispersion_ratio_vs_recorded": main},
            }
        }
        (image_swap_dir / f"{step:06d}.json").write_text(json.dumps(report))

    step, loss, _ = find_best_checkpoint(
        output_dir,
        image_swap_dir=image_swap_dir,
        min_paired_ratio=0.50,
        min_main_ratio=0.45,
    )

    assert step == 4000
    assert loss == pytest.approx(0.09)


def test_best_act_checkpoint_accepts_single_camera_all_images_report(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    image_swap_dir = output_dir / "image_swap"
    image_swap_dir.mkdir(parents=True)
    (output_dir / "train.log").write_text("step 1000: eval_loss=0.2000\n")
    (output_dir / "checkpoints" / "001000" / "pretrained_model").mkdir(
        parents=True
    )
    report = {
        "predicted_action_dispersion": {
            "all_images": {"dispersion_ratio_vs_recorded": 0.40},
            "main_only": {"dispersion_ratio_vs_recorded": 0.40},
        }
    }
    (image_swap_dir / "001000.json").write_text(json.dumps(report))

    step, loss, _ = find_best_checkpoint(
        output_dir,
        image_swap_dir=image_swap_dir,
        min_paired_ratio=0.30,
        min_main_ratio=0.30,
    )

    assert step == 1000
    assert loss == pytest.approx(0.20)


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
