from __future__ import annotations

import argparse
from pathlib import Path

import torch
from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors

from .configuration_v11 import V11Config
from .modeling_v11 import V11Policy
from .processor_v11 import make_v11_pre_post_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a trained V11 checkpoint to a LeRobot policy directory."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-source-root", type=Path, required=True)
    parser.add_argument("--tcc-real-robot-source-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def export_checkpoint(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise TypeError("Expected a V11 training checkpoint dictionary")
    policy_cfg = checkpoint["config"].get("policy", {})
    expected = {
        "implementation": "tcc_mlp_bc_v11_basic_chunked_absolute",
        "architecture": "pooled_feature_mlp",
        "cameras": ["cam_main"],
        "proprioception": True,
        "action_representation": "absolute",
        "action_space": "joint_position",
        "action_chunk_size": 40,
        "action_steps_per_inference": 10,
    }
    mismatches = {
        key: (expected_value, policy_cfg.get(key))
        for key, expected_value in expected.items()
        if policy_cfg.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint is not the supported V11 contract: {mismatches}")
    config = V11Config(
        device=args.device,
        backbone_checkpoint=str(args.backbone_checkpoint.resolve()),
        backbone_source_root=str(args.backbone_source_root.resolve()),
        tcc_real_robot_source_root=str(args.tcc_real_robot_source_root.resolve()),
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dimensions=tuple(policy_cfg["hidden_dimensions"]),
        action_dim=int(policy_cfg["action_dim"]),
        action_chunk_size=int(policy_cfg["action_chunk_size"]),
        n_action_steps=int(policy_cfg["action_steps_per_inference"]),
        proprioception_dim=int(policy_cfg["proprioception_dim"]),
        number_of_tasks=int(policy_cfg["number_of_tasks"]),
        input_features={
            "observation.images.cam_main": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            ),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
        },
    )
    policy = V11Policy(config)
    policy.head.load_state_dict(checkpoint["model"], strict=True)
    for name in ("action_mean", "action_std", "state_mean", "state_std"):
        value = torch.as_tensor(checkpoint[name], dtype=torch.float32)
        target = getattr(policy, name)
        if value.shape != target.shape:
            raise ValueError(f"{name} shape mismatch: {tuple(value.shape)} != {tuple(target.shape)}")
        target.copy_(value)
    if checkpoint.get("backbone_model") is not None:
        policy.backbone.load_state_dict(checkpoint["backbone_model"], strict=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    preprocessor, postprocessor = make_v11_pre_post_processors(config)
    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)

    # Exercise the same strict load and processor path used by lerobot-rollout
    # before the official runtime is allowed to connect to physical hardware.
    restored_config = PreTrainedConfig.from_pretrained(args.output_dir)
    restored_policy = V11Policy.from_pretrained(
        args.output_dir, config=restored_config, strict=True
    )
    make_pre_post_processors(
        restored_config,
        pretrained_path=str(args.output_dir),
    )
    synthetic_observation = {
        "observation.images.cam_main": torch.zeros(
            (1, 3, 480, 640), dtype=torch.float32
        ),
        "observation.state": torch.zeros((1, 7), dtype=torch.float32),
    }
    synthetic_action = restored_policy.select_action(synthetic_observation)
    if synthetic_action.shape != (1, 7) or not torch.isfinite(synthetic_action).all():
        raise RuntimeError("Exported V11 policy failed its synthetic action check")
    print("V11 LeRobot export self-check: PASS")


def main() -> None:
    export_checkpoint(parse_args())


if __name__ == "__main__":
    main()
