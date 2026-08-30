from __future__ import annotations

from lerobot.policies.act.modeling_act import ACTPolicy
from torch import nn

from .configuration_layerwise_act import ACTRN50LayerwiseConfig


class ACTRN50LayerwisePolicy(ACTPolicy):
    """Official ACT whose RN50 is optimized with two depth-aware LR groups."""

    config_class = ACTRN50LayerwiseConfig
    name = "act_rn50_layerwise"

    def get_optim_params(self) -> list[dict[str, object]]:
        policy_and_projection: list[nn.Parameter] = []
        backbone_early: list[nn.Parameter] = []
        backbone_late: list[nn.Parameter] = []

        early_prefixes = (
            "model.backbone.conv1.",
            "model.backbone.bn1.",
            "model.backbone.layer1.",
            "model.backbone.layer2.",
        )
        late_prefixes = (
            "model.backbone.layer3.",
            "model.backbone.layer4.",
        )

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(early_prefixes):
                backbone_early.append(parameter)
            elif name.startswith(late_prefixes):
                backbone_late.append(parameter)
            elif name.startswith("model.backbone."):
                raise RuntimeError(f"Unassigned RN50 backbone parameter: {name}")
            else:
                policy_and_projection.append(parameter)

        if not policy_and_projection or not backbone_early or not backbone_late:
            raise RuntimeError(
                "Expected non-empty ACT/projection, RN50 early, and RN50 late parameter groups"
            )

        return [
            {
                "params": policy_and_projection,
                "lr": self.config.optimizer_lr,
                "name": "act_and_projection",
            },
            {
                "params": backbone_late,
                "lr": self.config.optimizer_lr_backbone,
                "name": "backbone_layer3_layer4",
            },
            {
                "params": backbone_early,
                "lr": self.config.optimizer_lr_backbone_early,
                "name": "backbone_stem_layer1_layer2",
            },
        ]
