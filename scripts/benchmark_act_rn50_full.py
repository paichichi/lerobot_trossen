"""Measure full-resolution dual-camera ACT RN50 training throughput on CUDA."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot_policy_backbone_act import ACTRN50FullConfig
from lerobot_policy_backbone_act.modeling_native_rn50_act import ACTRN50FullPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-source-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--channels-last", action="store_true")
    return parser.parse_args()


def make_policy(args: argparse.Namespace) -> ACTRN50FullPolicy:
    os.environ["BACKBONE_CHECKPOINT"] = str(args.backbone_checkpoint.resolve())
    os.environ["BACKBONE_SOURCE_ROOT"] = str(args.backbone_source_root.resolve())
    config = ACTRN50FullConfig(
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
    policy = ACTRN50FullPolicy(config).cuda().train()
    if args.channels_last:
        policy = policy.to(memory_format=torch.channels_last)
    return policy


def make_batch(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    image_memory_format = (
        torch.channels_last if args.channels_last else torch.contiguous_format
    )

    def image() -> torch.Tensor:
        return torch.rand(
            args.batch_size, 3, 480, 640, dtype=torch.float32, device="cuda"
        ).contiguous(memory_format=image_memory_format)

    return {
        "observation.state": torch.randn(args.batch_size, 7, device="cuda"),
        "observation.images.cam_main": image(),
        "observation.images.cam_wrist": image(),
        "action": torch.randn(args.batch_size, 40, 7, device="cuda"),
        "action_is_pad": torch.zeros(
            args.batch_size, 40, dtype=torch.bool, device="cuda"
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    policy = make_policy(args)
    batch = make_batch(args)
    backbone_parameters = list(policy.model.backbone.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in policy.parameters() if id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": 1e-4},
            {"params": backbone_parameters, "lr": 1e-5},
        ],
        fused=True,
    )

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = [step() for _ in range(args.measure_steps)]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "gpu": torch.cuda.get_device_name(),
        "batch_size": args.batch_size,
        "image_shape": [480, 640],
        "cameras": 2,
        "precision": "bf16",
        "channels_last": args.channels_last,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "seconds": elapsed,
        "steps_per_second": args.measure_steps / elapsed,
        "samples_per_second": args.batch_size * args.measure_steps / elapsed,
        "milliseconds_per_step": elapsed * 1000 / args.measure_steps,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "last_loss": losses[-1],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
