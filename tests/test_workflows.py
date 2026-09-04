from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_repository_exposes_only_training_commands() -> None:
    scripts = {path.name for path in (ROOT / "scripts").iterdir() if path.is_file()}
    assert scripts == {"practice.sh", "train_policy.sh", "train_all_policies.sh"}


def test_training_matrix_is_three_by_three() -> None:
    script = (ROOT / "scripts/train_all_policies.sh").read_text()
    assert "open_lid push_pot press_button" in script
    assert "ours_vit hrp_imagenet d4r_imagenet" in script


def test_training_keeps_the_fixed_contract() -> None:
    script = (ROOT / "scripts/train_policy.sh").read_text()
    for value in (
        "--policy.freeze_vision_backbone=true",
        "--steps=\"$steps\"",
        "--save_freq=2000",
        "--dataset.image_transforms.max_num_transforms=3",
    ):
        assert value in script
