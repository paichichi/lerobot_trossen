#!/usr/bin/env python3
"""Measure whether an ACT checkpoint changes its actions when only images change.

This is an offline diagnostic. It never connects to or commands a robot.
For every validation episode, it fixes proprioception to one reference home
state and varies the paired main/wrist first-frame images. It also varies each
camera independently to expose which view drives the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_train_eval_datasets
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot_policy_backbone_act import ACTRN50FullConfig, BackboneACTConfig
from lerobot_policy_backbone_act.modeling_backbone_act import BackboneACTPolicy
from lerobot_policy_backbone_act.modeling_native_rn50_act import ACTRN50FullPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--video-backend",
        choices=("torchcodec", "pyav"),
        help="Override the checkpoint dataset video decoder for offline diagnostics.",
    )
    return parser.parse_args()


def as_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def dispersion(actions: np.ndarray) -> dict[str, object]:
    """Report cross-observation action dispersion in physical output units."""
    centered = actions - actions.mean(axis=0, keepdims=True)
    flat = centered.reshape(centered.shape[0], -1)
    return {
        "per_dimension_std": actions.std(axis=0).mean(axis=0).tolist(),
        "rms_about_mean": float(np.sqrt(np.mean(centered**2))),
        "mean_l2_about_mean": float(np.linalg.norm(flat, axis=1).mean()),
        "max_l2_about_mean": float(np.linalg.norm(flat, axis=1).max()),
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    cfg = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.policy.device = args.device
    if args.video_backend is not None:
        cfg.dataset.video_backend = args.video_backend
    _, eval_dataset = make_train_eval_datasets(cfg)
    if eval_dataset is None:
        raise RuntimeError("Checkpoint training config has no validation split")

    if isinstance(cfg.policy, ACTRN50FullConfig):
        policy_class = ACTRN50FullPolicy
    elif isinstance(cfg.policy, BackboneACTConfig):
        policy_class = BackboneACTPolicy
    else:
        policy_class = ACTPolicy
    policy = policy_class.from_pretrained(
        checkpoint, config=cfg.policy, local_files_only=True
    )
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    frame_indices = eval_dataset.hf_dataset["frame_index"]
    first_rows = [index for index, frame in enumerate(frame_indices) if int(frame) == 0]
    samples = [eval_dataset[index] for index in first_rows]
    if len(samples) != len(eval_dataset.episodes):
        raise RuntimeError(
            f"Expected {len(eval_dataset.episodes)} first frames, found {len(samples)}"
        )

    camera_keys = list(eval_dataset.meta.camera_keys)
    if len(camera_keys) != 2:
        raise RuntimeError(f"Expected two cameras, found {camera_keys}")
    main_key = next(key for key in camera_keys if key.endswith("cam_main"))
    wrist_key = next(key for key in camera_keys if key.endswith("cam_wrist"))
    reference = samples[0]

    def predict(sample: dict[str, object], main_image: object, wrist_image: object) -> np.ndarray:
        observation = {
            key: value
            for key, value in sample.items()
            if key != "action" and not key.startswith("observation.images.")
        }
        # Hold every state-like input fixed; vary only the requested images.
        for key, value in reference.items():
            if key.startswith("observation.") and not key.startswith("observation.images."):
                observation[key] = value
        observation[main_key] = main_image
        observation[wrist_key] = wrist_image
        for key in camera_keys:
            image = observation[key]
            if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
                observation[key] = image.to(dtype=torch.float32) / 255.0
        processed = preprocessor(observation)
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(processed)
            predicted = postprocessor(normalized)
        return as_tensor(predicted).squeeze(0).numpy()

    scenarios: dict[str, list[np.ndarray]] = {
        "paired_images": [],
        "main_only": [],
        "wrist_only": [],
    }
    recorded: list[np.ndarray] = []
    episodes: list[int] = []
    for sample in samples:
        episodes.append(int(sample["episode_index"]))
        recorded.append(as_tensor(sample["action"]).numpy())
        scenarios["paired_images"].append(
            predict(sample, sample[main_key], sample[wrist_key])
        )
        scenarios["main_only"].append(
            predict(sample, sample[main_key], reference[wrist_key])
        )
        scenarios["wrist_only"].append(
            predict(sample, reference[main_key], sample[wrist_key])
        )

    n_steps = min(policy.config.n_action_steps, scenarios["paired_images"][0].shape[0])
    recorded_array = np.stack(recorded)[:, :n_steps]
    recorded_stats = dispersion(recorded_array)
    scenario_stats: dict[str, object] = {}
    for name, values in scenarios.items():
        array = np.stack(values)[:, :n_steps]
        stats = dispersion(array)
        stats["dispersion_ratio_vs_recorded"] = float(
            stats["rms_about_mean"] / max(recorded_stats["rms_about_mean"], 1e-12)
        )
        rolled = np.roll(array, shift=1, axis=0)
        stats["mean_l2_change_after_episode_image_swap"] = float(
            np.linalg.norm((array - rolled).reshape(array.shape[0], -1), axis=1).mean()
        )
        scenario_stats[name] = stats

    paired_ratio = scenario_stats["paired_images"]["dispersion_ratio_vs_recorded"]
    if paired_ratio < 0.1:
        interpretation = "SEVERE_VISUAL_COLLAPSE"
    elif paired_ratio < 0.3:
        interpretation = "WEAK_VISUAL_CONDITIONING"
    else:
        interpretation = "MEANINGFUL_VISUAL_CONDITIONING"

    report = {
        "checkpoint": str(checkpoint),
        "device": args.device,
        "validation_episodes": episodes,
        "reference_episode": episodes[0],
        "fixed_inputs": ["observation.state", "observation.cartesian_position"],
        "varied_inputs": {
            "paired_images": [main_key, wrist_key],
            "main_only": [main_key],
            "wrist_only": [wrist_key],
        },
        "evaluated_action_steps": n_steps,
        "recorded_action_dispersion": recorded_stats,
        "predicted_action_dispersion": scenario_stats,
        "interpretation": interpretation,
        "notes": [
            "Both cameras are swapped as a synchronized pair in paired_images.",
            "The ratio compares policy output diversity with recorded action diversity.",
            "Thresholds are diagnostic heuristics, not task-success guarantees.",
        ],
    }
    output = args.output or checkpoint.parent.parent.parent / "image_swap_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
