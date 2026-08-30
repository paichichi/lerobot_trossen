from lerobot.policies.act.processor_act import make_act_pre_post_processors

from .configuration_late_act import ACTRN50LateConfig


def make_act_rn50_late_pre_post_processors(
    config: ACTRN50LateConfig, dataset_stats=None
):
    """Use the unmodified official ACT preprocessing and postprocessing."""

    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
