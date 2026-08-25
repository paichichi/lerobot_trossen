#!/usr/bin/env python3
"""Offline first-frame safety report for an ACT checkpoint.

The report loads the checkpoint's saved pre/post-processors, evaluates frame 0
of every validation episode, and compares predicted joint-position commands
with the current recorded robot state. No robot or environment is commanded.
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
from lerobot_policy_backbone_act import BackboneACTConfig, NativeRN50ACTConfig
from lerobot_policy_backbone_act.modeling_backbone_act import BackboneACTPolicy
from lerobot_policy_backbone_act.modeling_native_rn50_act import NativeRN50ACTPolicy


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--driver-arm-limit-rad", type=float, default=0.07)
    parser.add_argument("--driver-gripper-limit-m", type=float, default=0.003)
    parser.add_argument("--recorded-envelope-multiplier", type=float, default=2.0)
    return parser.parse_args()


def _tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def main() -> None:
    args = _args()
    checkpoint = args.checkpoint.resolve()

    cfg = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True)
    cfg.policy.device = args.device
    _, eval_dataset = make_train_eval_datasets(cfg)
    if eval_dataset is None:
        raise RuntimeError("Checkpoint training config has no validation split")

    if isinstance(cfg.policy, NativeRN50ACTConfig):
        policy_class = NativeRN50ACTPolicy
    elif isinstance(cfg.policy, BackboneACTConfig):
        policy_class = BackboneACTPolicy
    else:
        policy_class = ACTPolicy
    policy = policy_class.from_pretrained(
        checkpoint,
        config=cfg.policy,
        local_files_only=True,
    )
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    frame_indices = eval_dataset.hf_dataset["frame_index"]
    first_rows = [i for i, frame in enumerate(frame_indices) if int(frame) == 0]
    if len(first_rows) != len(eval_dataset.episodes):
        raise RuntimeError(
            f"Expected {len(eval_dataset.episodes)} first frames, found {len(first_rows)}"
        )

    rows: list[dict[str, object]] = []
    for dataset_index in first_rows:
        sample = eval_dataset[dataset_index]
        episode = int(sample["episode_index"])
        state = _tensor(sample["observation.state"]).reshape(-1)
        recorded_chunk = _tensor(sample["action"])
        if recorded_chunk.ndim == 1:
            recorded_chunk = recorded_chunk.unsqueeze(0)

        # Deployment never has the future action. Keeping it in the batch would
        # make ACT use the VAE posterior (training path) instead of its prior.
        observation_only = {key: value for key, value in sample.items() if key != "action"}
        # Match lerobot_train.py exactly: decoded RGB is uint8 for efficient
        # loading, then converted to [0, 1] float before the policy processor.
        for camera_key in eval_dataset.meta.camera_keys:
            image = observation_only.get(camera_key)
            if isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
                observation_only[camera_key] = image.to(dtype=torch.float32) / 255.0
        processed = preprocessor(observation_only)
        with torch.inference_mode():
            normalized_chunk = policy.predict_action_chunk(processed)
            predicted_chunk = _tensor(postprocessor(normalized_chunk)).squeeze(0)

        predicted_delta = predicted_chunk[0] - state
        recorded_delta = recorded_chunk[0] - state
        queued_steps = min(policy.config.n_action_steps, predicted_chunk.shape[0])
        queued = predicted_chunk[:queued_steps]
        queued_delta = torch.cat(
            [queued[:1] - state.unsqueeze(0), queued[1:] - queued[:-1]], dim=0
        )
        rows.append(
            {
                "episode": episode,
                "predicted_first": predicted_chunk[0].tolist(),
                "state": state.tolist(),
                "recorded_first": recorded_chunk[0].tolist(),
                "predicted_minus_state": predicted_delta.tolist(),
                "recorded_minus_state": recorded_delta.tolist(),
                "predicted_arm_abs_max_rad": float(predicted_delta[:6].abs().max()),
                "predicted_gripper_abs_m": float(predicted_delta[6].abs()),
                "recorded_arm_abs_max_rad": float(recorded_delta[:6].abs().max()),
                "recorded_gripper_abs_m": float(recorded_delta[6].abs()),
                "queued_arm_step_abs_max_rad": float(queued_delta[:, :6].abs().max()),
                "queued_gripper_step_abs_max_m": float(queued_delta[:, 6].abs().max()),
            }
        )

    pred_delta = np.asarray([row["predicted_minus_state"] for row in rows], dtype=np.float64)
    rec_delta = np.asarray([row["recorded_minus_state"] for row in rows], dtype=np.float64)
    pred_arm = np.asarray([row["predicted_arm_abs_max_rad"] for row in rows])
    pred_gripper = np.asarray([row["predicted_gripper_abs_m"] for row in rows])
    rec_arm = np.asarray([row["recorded_arm_abs_max_rad"] for row in rows])
    rec_gripper = np.asarray([row["recorded_gripper_abs_m"] for row in rows])
    queue_arm = np.asarray([row["queued_arm_step_abs_max_rad"] for row in rows])
    queue_gripper = np.asarray([row["queued_gripper_step_abs_max_m"] for row in rows])

    observed_arm_gate = args.recorded_envelope_multiplier * float(rec_arm.max())
    observed_gripper_gate = args.recorded_envelope_multiplier * float(rec_gripper.max())
    arm_gate = min(args.driver_arm_limit_rad, observed_arm_gate)
    gripper_gate = min(args.driver_gripper_limit_m, observed_gripper_gate)
    passed = bool(pred_arm.max() <= arm_gate and pred_gripper.max() <= gripper_gate)

    report = {
        "checkpoint": str(checkpoint),
        "device": args.device,
        "validation_episodes": [int(x) for x in eval_dataset.episodes],
        "num_first_frames": len(rows),
        "units": {"joints_0_to_5": "rad", "gripper_6": "m"},
        "summary": {
            "predicted_arm_abs_max_rad": float(pred_arm.max()),
            "predicted_arm_abs_median_rad": float(np.median(pred_arm)),
            "recorded_arm_abs_max_rad": float(rec_arm.max()),
            "recorded_arm_abs_median_rad": float(np.median(rec_arm)),
            "predicted_gripper_abs_max_m": float(pred_gripper.max()),
            "predicted_gripper_abs_median_m": float(np.median(pred_gripper)),
            "recorded_gripper_abs_max_m": float(rec_gripper.max()),
            "recorded_gripper_abs_median_m": float(np.median(rec_gripper)),
            "predicted_minus_state_signed_mean": pred_delta.mean(axis=0).tolist(),
            "recorded_minus_state_signed_mean": rec_delta.mean(axis=0).tolist(),
            "queued_first_10_arm_step_abs_max_rad": float(queue_arm.max()),
            "queued_first_10_gripper_step_abs_max_m": float(queue_gripper.max()),
        },
        "gates": {
            "driver_arm_limit_rad": args.driver_arm_limit_rad,
            "driver_gripper_limit_m": args.driver_gripper_limit_m,
            "recorded_envelope_multiplier": args.recorded_envelope_multiplier,
            "effective_arm_gate_rad": arm_gate,
            "effective_gripper_gate_m": gripper_gate,
        },
        "decision": "PASS" if passed else "BLOCKED",
        "episodes": rows,
    }
    output = args.output or checkpoint.parent.parent.parent / "first_frame_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"report={output}")


if __name__ == "__main__":
    main()
