from lerobot.policies.act.processor_act import make_act_pre_post_processors

from .configuration_layerwise_act import ACTRN50LayerwiseConfig


def make_act_rn50_layerwise_pre_post_processors(
    config: ACTRN50LayerwiseConfig, dataset_stats=None
):
    """Use the unmodified official ACT preprocessing and postprocessing."""

    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
