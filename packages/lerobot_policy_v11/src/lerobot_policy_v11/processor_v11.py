from typing import Any

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)

from .configuration_v11 import V11Config


def make_v11_pre_post_processors(
    config: V11Config,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Use only LeRobot's standard batching/device processors.

    V11 checkpoints already contain their training-state and action statistics,
    while visual ImageNet normalization belongs to the frozen backbone adapter.
    """

    del dataset_stats
    preprocessor = PolicyProcessorPipeline(
        steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=config.device),
        ],
        name="policy_preprocessor",
    )
    postprocessor = PolicyProcessorPipeline[
        PolicyAction, PolicyAction
    ](
        steps=[DeviceProcessorStep(device="cpu")],
        name="policy_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor
