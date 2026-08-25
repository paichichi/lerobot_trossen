from lerobot.policies.act.processor_act import make_act_pre_post_processors

from .configuration_native_rn50_act import ACTRN50FullConfig


def make_act_rn50_full_pre_post_processors(
    config: ACTRN50FullConfig, dataset_stats=None
):
    """Use the unmodified official ACT state/action processing pipeline."""

    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
