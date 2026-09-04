# Frozen-ViT ACT Training

This repository trains official LeRobot ACT with a frozen ViT-B/16 image
encoder. It supports three tasks and three backbones.

Tasks:

- `open_lid`
- `push_pot`
- `press_button`

Backbones:

- `ours_vit`
- `hrp_imagenet`
- `d4r_imagenet`

## Install

```bash
uv sync --extra act
```

## Train

Train one policy:

```bash
bash scripts/train_policy.sh press_button ours_vit
```

Train all nine policies sequentially:

```bash
bash scripts/train_all_policies.sh
```

## Practice

Practice leader-to-follower teleoperation without recording or cameras:

```bash
bash scripts/practice.sh
```

Keep the E-stop ready. Both arms move to their staged positions when practice
starts and move to sleep positions during normal shutdown.

Each run trains for 8,000 steps and saves checkpoints at 2K, 4K, 6K, and 8K.
Training outputs are written to `outputs/train/` and are not committed to Git.

## Required local files

```text
assets/tcc-policy-assets/backbones/
|-- ours_vit/checkpoint_040000.pt
|-- hrp_imagenet/HRP_IN.pth
`-- d4r_imagenet/D4R_IN_1M.pth

datasets/
|-- open_the_pot_by_lifting_the_lid/dataset/
|-- push_the_pot_into_the_marked_area/dataset/
`-- press_the_button/dataset/
```

The datasets, backbone weights, caches, logs, and training outputs are local
artifacts and are ignored by Git.
