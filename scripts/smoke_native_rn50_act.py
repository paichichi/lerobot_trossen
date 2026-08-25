"""GPU forward/backward smoke test for native 480x640 full ACT RN50."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot_policy_backbone_act import NativeRN50ACTConfig
from lerobot_policy_backbone_act.modeling_native_rn50_act import NativeRN50ACTPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-source-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native RN50 ACT smoke test")
    os.environ["BACKBONE_CHECKPOINT"] = str(args.backbone_checkpoint.resolve())
    os.environ["BACKBONE_SOURCE_ROOT"] = str(args.backbone_source_root.resolve())

    config = NativeRN50ACTConfig(
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
        backbone_checkpoint=str(args.backbone_checkpoint),
        backbone_source_root=str(args.backbone_source_root),
        chunk_size=40,
        n_action_steps=10,
    )
    policy = NativeRN50ACTPolicy(config).cuda().train()
    batch = {
        "observation.state": torch.randn(args.batch_size, 7, device="cuda"),
        "observation.images.cam_main": torch.randint(
            0,
            256,
            (args.batch_size, 3, 480, 640),
            dtype=torch.uint8,
            device="cuda",
        ),
        "observation.images.cam_wrist": torch.randint(
            0,
            256,
            (args.batch_size, 3, 480, 640),
            dtype=torch.uint8,
            device="cuda",
        ),
        "action": torch.randn(args.batch_size, 40, 7, device="cuda"),
        "action_is_pad": torch.zeros(
            args.batch_size, 40, dtype=torch.bool, device="cuda"
        ),
    }

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, loss_dict = policy(batch)
    loss.backward()
    backbone_gradients = [
        parameter.grad
        for parameter in policy.model.backbone.backbone.parameters()
        if parameter.requires_grad
    ]
    report = {
        "batch_size": args.batch_size,
        "image_shape": [480, 640],
        "dim_model": config.dim_model,
        "n_heads": config.n_heads,
        "dim_feedforward": config.dim_feedforward,
        "n_encoder_layers": config.n_encoder_layers,
        "n_vae_encoder_layers": config.n_vae_encoder_layers,
        "loss": float(loss.detach()),
        "losses": {
            key: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
            for key, value in loss_dict.items()
        },
        "backbone_trainable": all(
            parameter.requires_grad
            for parameter in policy.model.backbone.backbone.parameters()
        ),
        "backbone_has_finite_gradient": any(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in backbone_gradients
        ),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
