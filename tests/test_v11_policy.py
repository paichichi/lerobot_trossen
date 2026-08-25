from pathlib import Path

import pytest
import torch
from lerobot_policy_v11 import modeling_v11
from lerobot_policy_v11.configuration_v11 import V11Config
from lerobot_policy_v11.export_v11 import (
    EXPECTED_POLICY_CONTRACT,
    validate_policy_contract,
)
from lerobot_policy_v11.processor_v11 import make_v11_pre_post_processors
from torch import nn


class FakeBackbone(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.ones((image.shape[0], 4), device=image.device)


def make_policy(monkeypatch) -> modeling_v11.V11Policy:
    monkeypatch.setattr(
        modeling_v11, "_load_upstream_backbone", lambda config: FakeBackbone()
    )
    tcc_root = Path(__file__).parents[2] / "tcc-core-real-robot"
    config = V11Config(
        device="cpu",
        tcc_real_robot_source_root=str(tcc_root),
        feature_dim=4,
        hidden_dimensions=(8,),
    )
    policy = modeling_v11.V11Policy(config)
    for parameter in policy.head.parameters():
        parameter.data.zero_()
    policy.action_mean.copy_(torch.arange(280, dtype=torch.float32))
    return policy


def test_v11_adapter_preserves_checkpoint_chunk_and_queue(monkeypatch) -> None:
    policy = make_policy(monkeypatch)
    batch = {
        "observation.images.cam_main": torch.zeros(
            (1, 3, 8, 8), dtype=torch.uint8
        ),
        "observation.state": torch.zeros((1, 7)),
    }

    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (1, 40, 7)
    torch.testing.assert_close(chunk[0, 0], torch.arange(7, dtype=torch.float32))
    torch.testing.assert_close(chunk[0, 9], torch.arange(63, 70, dtype=torch.float32))

    for step in range(10):
        torch.testing.assert_close(
            policy.select_action(batch)[0],
            torch.arange(step * 7, step * 7 + 7, dtype=torch.float32),
        )
    torch.testing.assert_close(
        policy.select_action(batch)[0], torch.arange(7, dtype=torch.float32)
    )


def test_v11_processors_leave_checkpoint_normalization_in_policy() -> None:
    config = V11Config(device="cpu")
    preprocessor, postprocessor = make_v11_pre_post_processors(config)
    assert len(preprocessor.steps) == 3
    assert len(postprocessor.steps) == 1


@pytest.mark.parametrize(
    "implementation",
    [
        "tcc_mlp_bc_v11_basic_chunked_absolute",
        "tcc_mlp_bc_v11_end_to_end_chunked_absolute",
    ],
)
def test_v11_export_accepts_frozen_and_end_to_end_training_contracts(
    implementation: str,
) -> None:
    policy_cfg = {**EXPECTED_POLICY_CONTRACT, "implementation": implementation}

    validate_policy_contract(policy_cfg)


def test_v11_export_still_rejects_incompatible_action_contract() -> None:
    policy_cfg = {
        **EXPECTED_POLICY_CONTRACT,
        "implementation": "tcc_mlp_bc_v11_end_to_end_chunked_absolute",
        "action_representation": "residual",
    }

    with pytest.raises(RuntimeError, match="action_representation"):
        validate_policy_contract(policy_cfg)
