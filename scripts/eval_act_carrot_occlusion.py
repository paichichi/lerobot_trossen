#!/usr/bin/env python3
"""Offline ACT diagnostic that compares carrot and empty-table occlusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args()


def image_to_hwc_uint8(image: torch.Tensor) -> tuple[np.ndarray, bool]:
    array = image.detach().cpu().numpy()
    channel_first = array.ndim == 3 and array.shape[0] == 3
    if channel_first:
        array = np.moveaxis(array, 0, -1)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array, channel_first


def restore_image(array: np.ndarray, original: torch.Tensor, channel_first: bool) -> torch.Tensor:
    if channel_first:
        array = np.moveaxis(array, -1, 0)
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if original.dtype != torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    return tensor.to(dtype=original.dtype)


def carrot_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # The carrot body is the only highly saturated orange object in this dataset.
    mask = cv2.inRange(hsv, np.array([3, 150, 150]), np.array([24, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [stats[index] for index in range(1, count) if stats[index, cv2.CC_STAT_AREA] >= 150]
    if not candidates:
        raise RuntimeError("Could not locate the orange carrot")
    x, y, width, height, _ = max(candidates, key=lambda item: item[cv2.CC_STAT_AREA])
    padding = 10
    return (
        max(0, int(x) - padding),
        max(0, int(y) - padding),
        min(rgb.shape[1], int(x + width) + padding),
        min(rgb.shape[0], int(y + height) + padding),
    )


def table_fill(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    table = rgb[(hsv[..., 1] < 45) & (hsv[..., 2] > 80)]
    if len(table) == 0:
        return np.array([150, 145, 140], dtype=np.uint8)
    return np.median(table, axis=0).astype(np.uint8)


def occlude(rgb: np.ndarray, box: tuple[int, int, int, int], fill: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    x0, y0, x1, y1 = box
    result[y0:y1, x0:x1] = fill
    return result


def main() -> None:
    options = args()
    checkpoint = options.checkpoint.resolve()
    cfg = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.policy.device = options.device
    cfg.dataset.root = options.dataset_root.resolve()
    cfg.dataset.video_backend = options.video_backend
    train_dataset, eval_dataset = make_train_eval_datasets(cfg)
    dataset = eval_dataset if eval_dataset is not None else train_dataset

    policy = ACTPolicy.from_pretrained(checkpoint, config=cfg.policy, local_files_only=True)
    policy.to(options.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": options.device}},
    )
    camera_key = next(key for key in dataset.meta.camera_keys if key.endswith("cam_main"))
    first_rows = [
        index for index, frame in enumerate(dataset.hf_dataset["frame_index"]) if int(frame) == 0
    ]
    samples = [dataset[index] for index in first_rows]
    reference = samples[0]

    def predict(sample: dict[str, object], image: torch.Tensor) -> np.ndarray:
        observation = {key: value for key, value in sample.items() if key != "action"}
        for key, value in reference.items():
            if key.startswith("observation.") and not key.startswith("observation.images."):
                observation[key] = value
        observation[camera_key] = image
        if image.dtype == torch.uint8:
            observation[camera_key] = image.to(torch.float32) / 255.0
        processed = preprocessor(observation)
        with torch.inference_mode():
            action = postprocessor(policy.predict_action_chunk(processed))
        return action.detach().cpu().float().squeeze(0).numpy()[: policy.config.n_action_steps]

    rows = []
    for sample in samples:
        original = sample[camera_key]
        rgb, channel_first = image_to_hwc_uint8(original)
        box = carrot_bbox(rgb)
        fill = table_fill(rgb)
        carrot_image = restore_image(occlude(rgb, box, fill), original, channel_first)
        width, height = box[2] - box[0], box[3] - box[1]
        # Fixed empty-table control patch, sized identically to the carrot patch.
        control_box = (20, 170, min(rgb.shape[1], 20 + width), min(rgb.shape[0], 170 + height))
        control_image = restore_image(occlude(rgb, control_box, fill), original, channel_first)
        full = predict(sample, original)
        carrot_hidden = predict(sample, carrot_image)
        control_hidden = predict(sample, control_image)
        rows.append(
            {
                "episode": int(sample["episode_index"]),
                "carrot_bbox_xyxy": list(box),
                "full_vs_carrot_l2": float(np.linalg.norm(full - carrot_hidden)),
                "full_vs_control_l2": float(np.linalg.norm(full - control_hidden)),
            }
        )

    carrot = np.array([row["full_vs_carrot_l2"] for row in rows])
    control = np.array([row["full_vs_control_l2"] for row in rows])
    report = {
        "checkpoint": str(checkpoint),
        "episodes": len(rows),
        "evaluated_action_steps": policy.config.n_action_steps,
        "mean_carrot_occlusion_l2": float(carrot.mean()),
        "median_carrot_occlusion_l2": float(np.median(carrot)),
        "mean_control_occlusion_l2": float(control.mean()),
        "carrot_vs_control_ratio": float(carrot.mean() / max(control.mean(), 1e-12)),
        "rows": rows,
        "note": "Training-split occlusion sensitivity is diagnostic, not held-out success.",
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    print(f"report={options.output}")


if __name__ == "__main__":
    main()
