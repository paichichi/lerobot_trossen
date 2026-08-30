from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_color3_changes_only_transform_count_in_the_training_recipe() -> None:
    baseline = (
        REPO_ROOT / "scripts/train_official_act_rn50_carrot_to_pot_40_color_lr1e6.sh"
    ).read_text()
    color3 = (
        REPO_ROOT / "scripts/train_official_act_rn50_carrot_to_pot_40_color3_lr1e6.sh"
    ).read_text()

    shared_arguments = (
        '--dataset.repo_id=UoA-Trossen-Arm/carrot_to_pot_40',
        '--dataset.image_transforms.enable=true',
        '--dataset.image_transforms.random_order=false',
        '--policy.type=act',
        '--policy.chunk_size=40',
        '--policy.n_action_steps=10',
        '--policy.vision_backbone=resnet50',
        '--policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1',
        '--policy.optimizer_lr=1e-5',
        '--policy.optimizer_lr_backbone=1e-6',
        '--policy.optimizer_weight_decay=1e-4',
        '--steps="$steps"',
    )
    for argument in shared_arguments:
        assert argument in baseline
        assert argument in color3

    assert '--dataset.image_transforms.max_num_transforms=2' in baseline
    assert '--dataset.image_transforms.max_num_transforms=3' in color3
    assert 'save_freq="${ACT_SAVE_FREQ:-1000}"' in color3
