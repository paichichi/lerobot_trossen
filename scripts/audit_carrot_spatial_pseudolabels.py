#!/usr/bin/env python3
"""Audit automatic carrot heatmaps without modifying the LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_backbone_act.modeling_native_rn50_act import (
    _automatic_carrot_heatmap_targets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LeRobotDataset(
        "UoA-Trossen-Arm/pick_and_place_carrot_100",
        root=args.dataset_root,
        video_backend="torchcodec",
        return_uint8=True,
    )
    first_rows = [
        index
        for index, frame in enumerate(dataset.hf_dataset["frame_index"])
        if int(frame) == 0
    ]
    episodes: list[dict[str, object]] = []
    for index in first_rows:
        sample = dataset[index]
        target, valid = _automatic_carrot_heatmap_targets(
            sample["observation.images.cam_main"].unsqueeze(0), (60, 80)
        )
        heatmap = target[0, 0]
        x_axis = torch.arange(80, dtype=heatmap.dtype)
        y_axis = torch.arange(60, dtype=heatmap.dtype)
        episodes.append(
            {
                "episode": int(sample["episode_index"]),
                "valid": bool(valid[0]),
                "center_x_px": float((heatmap.sum(0) * x_axis).sum() * 8 + 4),
                "center_y_px": float((heatmap.sum(1) * y_axis).sum() * 8 + 4),
            }
        )
    valid_count = sum(bool(item["valid"]) for item in episodes)
    valid_fraction = valid_count / max(len(episodes), 1)
    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "episodes": len(episodes),
        "valid": valid_count,
        "valid_fraction": valid_fraction,
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "labels_are_runtime_generated": True,
        "dataset_was_modified": False,
        "per_episode": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "per_episode"},
            indent=2,
        )
    )
    if valid_fraction < args.minimum_valid_fraction:
        raise RuntimeError(
            f"Automatic carrot pseudo-label coverage {valid_fraction:.1%} is below "
            f"the required {args.minimum_valid_fraction:.1%}"
        )


if __name__ == "__main__":
    main()
