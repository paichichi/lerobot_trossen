from __future__ import annotations

from lerobot.policies.act.modeling_act import ACTPolicy
from torch import nn

from .configuration_late_act import ACTRN50LateConfig


class ACTRN50LatePolicy(ACTPolicy):
    """Official ACT with a frozen RN50 stem/layer1/layer2."""

    config_class = ACTRN50LateConfig
    name = "act_rn50_late"

    _early_prefixes = (
        "model.backbone.conv1.",
        "model.backbone.bn1.",
        "model.backbone.layer1.",
        "model.backbone.layer2.",
    )
    _late_prefixes = (
        "model.backbone.layer3.",
        "model.backbone.layer4.",
    )

    def __init__(self, config: ACTRN50LateConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        for name, parameter in self.named_parameters():
            if name.startswith(self._early_prefixes):
                parameter.requires_grad_(False)

    def get_optim_params(self) -> list[dict[str, object]]:
        policy_and_projection: list[nn.Parameter] = []
        backbone_late: list[nn.Parameter] = []

        for name, parameter in self.named_parameters():
            if name.startswith(self._early_prefixes):
                if parameter.requires_grad:
                    raise RuntimeError(f"Early RN50 parameter was not frozen: {name}")
                continue
            if not parameter.requires_grad:
                continue
            if name.startswith(self._late_prefixes):
                backbone_late.append(parameter)
            elif name.startswith("model.backbone."):
                raise RuntimeError(f"Unassigned RN50 backbone parameter: {name}")
            else:
                policy_and_projection.append(parameter)

        if not policy_and_projection or not backbone_late:
            raise RuntimeError("Expected non-empty ACT/projection and RN50 layer3/layer4 groups")

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
        ]
