from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
VIT_SCRIPT = REPO_ROOT / "scripts/train_act_ours_vit_carrot_to_pot_40_color3.sh"
VIT_E2E_SCRIPT = (
    REPO_ROOT / "scripts/train_act_ours_vit_carrot_to_pot_40_color3_e2e.sh"
)
VIT_LAST1_SCRIPT = (
    REPO_ROOT / "scripts/train_act_ours_vit_carrot_to_pot_40_color3_last1.sh"
)
VIT_LAYERNORM_SCRIPT = (
    REPO_ROOT
    / "scripts/train_act_ours_vit_carrot_to_pot_40_color3_layernorm_only.sh"
)
VIT_LATER_SCRIPT = (
    REPO_ROOT / "scripts/train_act_ours_vit_carrot_to_pot_40_color3_later_only.sh"
)
VIT_COMPACT_SCRIPT = (
    REPO_ROOT / "scripts/train_act_ours_vit_carrot_to_pot_40_color3_compact_val.sh"
)
FROZEN_COMPARISON_SCRIPT = (
    REPO_ROOT / "scripts/train_act_frozen_backbone_comparison.sh"
)
COLOR3_SCRIPT = (
    REPO_ROOT / "scripts/train_official_act_rn50_carrot_to_pot_40_color3_lr1e6.sh"
)


def test_vit_training_uses_the_controlled_color3_dataset_contract() -> None:
    vit = VIT_SCRIPT.read_text()
    color3 = COLOR3_SCRIPT.read_text()

    shared_flags = (
        "--dataset.repo_id=UoA-Trossen-Arm/carrot_to_pot_40",
        "--dataset.eval_split=0.0",
        "--dataset.return_uint8=true",
        "--dataset.image_transforms.enable=true",
        "--dataset.image_transforms.max_num_transforms=3",
        "--dataset.image_transforms.random_order=false",
        "--dataset.image_transforms.tfs=\"$image_transforms\"",
        "--policy.chunk_size=40",
        "--policy.n_action_steps=10",
        "--eval_steps=0",
    )
    for flag in shared_flags:
        assert flag in vit
        assert flag in color3
    assert "source_config=\"$repo_root/checkpoints/act_official_rn50" in vit
    assert "if len(episodes) != 40" in vit


def test_vit_training_uses_frozen_tcc_patch_backbone() -> None:
    script = VIT_SCRIPT.read_text()

    assert "--policy.type=backbone_act" in script
    assert "--policy.backbone_family=ours_vit" in script
    assert "--policy.freeze_vision_backbone=true" in script
    assert "--policy.backbone_image_size=224" in script
    assert "--policy.optimizer_lr=1e-4" in script
    assert "--policy.discover_packages_path=lerobot_policy_backbone_act" in script
    assert "assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt" in script


def test_vit_training_saves_every_1000_steps_without_overwrite() -> None:
    script = VIT_SCRIPT.read_text()

    assert 'save_freq="${ACT_SAVE_FREQ:-1000}"' in script
    assert 'if [[ -e "$output_dir" || -e "$train_log" ]]' in script
    assert "Refusing to overwrite" in script


def test_vit_e2e_changes_only_the_optimization_contract() -> None:
    frozen = VIT_SCRIPT.read_text()
    e2e = VIT_E2E_SCRIPT.read_text()

    shared_flags = (
        "--dataset.repo_id=UoA-Trossen-Arm/carrot_to_pot_40",
        "--dataset.eval_split=0.0",
        "--dataset.image_transforms.max_num_transforms=3",
        "--dataset.image_transforms.random_order=false",
        "--dataset.image_transforms.tfs=\"$image_transforms\"",
        "--policy.type=backbone_act",
        "--policy.backbone_family=ours_vit",
        "--policy.backbone_image_size=224",
        "--policy.chunk_size=40",
        "--policy.n_action_steps=10",
        "--batch_size=\"$batch_size\"",
        "--eval_steps=0",
    )
    for flag in shared_flags:
        assert flag in frozen
        assert flag in e2e

    assert "--policy.freeze_vision_backbone=true" in frozen
    assert "--policy.freeze_vision_backbone=false" in e2e
    assert "--policy.optimizer_lr=1e-5" in e2e
    assert "--policy.optimizer_lr_backbone=1e-7" in e2e
    assert "color3_e2e_lr1e7" in e2e
    assert "color3_frozen" not in e2e


def test_vit_last_block_tuning_starts_from_frozen_8k_and_is_short() -> None:
    script = VIT_LAST1_SCRIPT.read_text()

    assert "color3_frozen_train40_8000steps" in script
    assert '--policy.path="$frozen_policy"' in script
    assert "--policy.discover_packages_path" not in script
    assert "--policy.freeze_vision_backbone=false" in script
    assert "--policy.vit_trainable_last_blocks=1" in script
    assert "--policy.optimizer_lr=1e-5" in script
    assert "--policy.optimizer_lr_backbone=1e-7" in script
    assert 'steps="${ACT_STEPS:-1000}"' in script
    assert 'save_freq="${ACT_SAVE_FREQ:-250}"' in script
    assert "--resume=true" not in script


def test_vit_layer_norm_only_trains_a_new_act_on_the_full_40() -> None:
    script = VIT_LAYERNORM_SCRIPT.read_text()

    assert '--policy.path="$frozen_policy"' not in script
    assert "--policy.discover_packages_path=lerobot_policy_backbone_act" in script
    assert "--policy.type=backbone_act" in script
    assert "if len(episodes) != 40" in script
    assert "--dataset.eval_split=0.0" in script
    assert "--dataset.image_transforms.max_num_transforms=3" in script
    assert "--policy.freeze_vision_backbone=false" in script
    assert "--policy.vit_train_layer_norm_only=true" in script
    assert "--policy.vit_trainable_last_blocks" not in script
    assert "--policy.optimizer_lr=1e-4" in script
    assert "--policy.optimizer_lr_backbone=1e-6" in script
    assert "--policy.optimizer_weight_decay=1e-4" in script
    assert 'steps="${ACT_STEPS:-8000}"' in script
    assert 'save_freq="${ACT_SAVE_FREQ:-1000}"' in script
    assert "--resume=true" not in script


def test_vit_later_only_differs_from_layer_norm_only_by_trainable_vit_set() -> None:
    layer_norm = VIT_LAYERNORM_SCRIPT.read_text()
    later = VIT_LATER_SCRIPT.read_text()

    shared_flags = (
        "--dataset.eval_split=0.0",
        "--dataset.image_transforms.max_num_transforms=3",
        "--policy.type=backbone_act",
        "--policy.freeze_vision_backbone=false",
        "--policy.optimizer_lr=1e-4",
        "--policy.optimizer_lr_backbone=1e-6",
        "--policy.optimizer_weight_decay=1e-4",
        '--batch_size="$batch_size"',
        '--steps="$steps"',
    )
    for flag in shared_flags:
        assert flag in layer_norm
        assert flag in later
    assert "--policy.vit_train_layer_norm_only=true" in layer_norm
    assert "--policy.vit_trainable_last_blocks=1" in later
    assert "--policy.vit_train_layer_norm_only" not in later


def test_frozen_backbone_comparison_uses_one_controlled_training_contract() -> None:
    script = FROZEN_COMPARISON_SCRIPT.read_text()

    for variant in ("ours_vit", "ours_rn50", "d4r_imagenet", "hrp_imagenet"):
        assert f"  {variant})" in script
    assert "backbone_family=pretrained_vit" in script
    assert "if len(episodes) != 40" in script
    assert "--dataset.eval_split=0.0" in script
    assert "--dataset.image_transforms.max_num_transforms=3" in script
    assert "--policy.freeze_vision_backbone=true" in script
    assert "--policy.optimizer_lr=1e-4" in script
    assert "--policy.optimizer_weight_decay=1e-4" in script
    assert 'steps="${ACT_STEPS:-8000}"' in script
    assert 'save_freq="${ACT_SAVE_FREQ:-2000}"' in script
    assert "Backbone hash mismatch" in script


def test_vit_compact_keeps_backbone_frozen_and_holds_out_validation() -> None:
    script = VIT_COMPACT_SCRIPT.read_text()

    assert "--policy.freeze_vision_backbone=true" in script
    assert "--policy.dim_model=256" in script
    assert "--policy.n_heads=4" in script
    assert "--policy.dim_feedforward=1024" in script
    assert "--policy.n_encoder_layers=2" in script
    assert "--policy.n_vae_encoder_layers=2" in script
    assert 'eval_split="${ACT_EVAL_SPLIT:-0.2}"' in script
    assert 'eval_steps="${ACT_EVAL_STEPS:-500}"' in script
    assert 'steps="${ACT_STEPS:-4000}"' in script
    assert "scripts/select_best_act_checkpoint.py" in script
