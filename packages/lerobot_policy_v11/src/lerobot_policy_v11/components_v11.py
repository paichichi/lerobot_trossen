from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional
from torchvision import models


class V11ResNet50Backbone(nn.Module):
    """Self-contained pooled RN50 matching the V11 checkpoint key contract."""

    output_dim = 2048

    def __init__(self) -> None:
        super().__init__()
        self.convnet = models.resnet50(weights=None)
        self.convnet.fc = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.convnet(images)


class V11MLPHead(nn.Module):
    """Single-view V11 head with state and one-hot task conditioning."""

    def __init__(
        self,
        *,
        feature_dim: int,
        number_of_tasks: int,
        proprioception_dim: int,
        hidden_dimensions: Sequence[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or number_of_tasks <= 0 or proprioception_dim <= 0:
            raise ValueError("V11 conditioning dimensions must be positive")
        if not hidden_dimensions or any(width <= 0 for width in hidden_dimensions):
            raise ValueError("V11 hidden dimensions must be positive")
        if output_dim <= 0:
            raise ValueError("V11 output dimension must be positive")

        layers: list[nn.Module] = []
        previous = feature_dim + number_of_tasks + proprioception_dim
        for width in hidden_dimensions:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.number_of_tasks = number_of_tasks
        self.proprioception_dim = proprioception_dim

    def forward(
        self,
        features: torch.Tensor,
        task_index: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("V11 features must be a two-dimensional tensor")
        if state.shape != (features.shape[0], self.proprioception_dim):
            raise ValueError("Unexpected V11 proprioception shape")
        if task_index.shape != (features.shape[0],):
            raise ValueError("Unexpected V11 task-index shape")
        task = functional.one_hot(
            task_index.long(), self.number_of_tasks
        ).to(features.dtype)
        return self.mlp(torch.cat((features, task, state), dim=-1))
