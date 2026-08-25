from __future__ import annotations

from collections import deque
from typing import Any

import torch
from lerobot.policies.pretrained import PreTrainedPolicy
from torch import Tensor, nn
from torch.nn import functional

from .components_v11 import V11MLPHead, V11ResNet50Backbone
from .configuration_v11 import V11Config


def _make_backbone(config: V11Config) -> nn.Module:
    backbone = V11ResNet50Backbone()
    for parameter in backbone.parameters():
        parameter.requires_grad_(not config.freeze_vision_backbone)
    return backbone


class V11Policy(PreTrainedPolicy):
    """Expose a trained V11 MLP through LeRobot's standard policy API."""

    config_class = V11Config
    name = "v11"

    def __init__(self, config: V11Config, **kwargs: Any) -> None:
        super().__init__(config)
        del kwargs
        self.backbone = _make_backbone(config)
        self.head = V11MLPHead(
            feature_dim=config.feature_dim,
            number_of_tasks=config.number_of_tasks,
            proprioception_dim=config.proprioception_dim,
            hidden_dimensions=config.hidden_dimensions,
            output_dim=config.action_dim * config.action_chunk_size,
        )
        flat_action_dim = config.action_dim * config.action_chunk_size
        self.register_buffer("action_mean", torch.zeros(flat_action_dim))
        self.register_buffer("action_std", torch.ones(flat_action_dim))
        self.register_buffer("state_mean", torch.zeros(config.proprioception_dim))
        self.register_buffer("state_std", torch.ones(config.proprioception_dim))
        self.register_buffer(
            "image_mean",
            torch.tensor(config.backbone_image_mean).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor(config.backbone_image_std).view(1, 3, 1, 1),
        )
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)

    def reset(self) -> None:
        self._action_queue.clear()

    def get_optim_params(self) -> dict:
        return self.head.parameters()

    def _preprocess_image(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW RGB input, got {tuple(image.shape)}")
        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32).div_(255.0)
        elif torch.is_floating_point(image):
            image = image.to(dtype=torch.float32)
            if image.numel() and float(image.detach().amax().cpu()) > 1.5:
                image = image.div(255.0)
        else:
            raise TypeError(f"Unsupported image dtype: {image.dtype}")
        image = functional.interpolate(
            image,
            size=(self.config.backbone_image_size, self.config.backbone_image_size),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        return (image - self.image_mean) / self.image_std

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        image_key = "observation.images.cam_main"
        state_key = "observation.state"
        if image_key not in batch or state_key not in batch:
            raise KeyError(f"V11 requires {image_key!r} and {state_key!r}")
        image = self._preprocess_image(batch[image_key])
        state = batch[state_key].to(dtype=torch.float32)
        if state.ndim != 2 or state.shape[1] != self.config.proprioception_dim:
            raise ValueError(f"Unexpected state shape: {tuple(state.shape)}")
        features = self.backbone(image).float()
        normalized_state = (state - self.state_mean) / self.state_std.clamp_min(1e-6)
        task = torch.full(
            (state.shape[0],),
            self.config.task_index,
            dtype=torch.long,
            device=state.device,
        )
        normalized_action = self.head(features, task, normalized_state)
        action = normalized_action * self.action_std + self.action_mean
        return action.reshape(
            state.shape[0], self.config.action_chunk_size, self.config.action_dim
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        prediction = self.predict_action_chunk(batch)
        if "action" not in batch:
            return {"action": prediction}
        target = batch["action"]
        if target.shape != prediction.shape:
            raise ValueError(
                f"Expected action target {tuple(prediction.shape)}, got {tuple(target.shape)}"
            )
        return {"loss": functional.mse_loss(prediction, target)}
