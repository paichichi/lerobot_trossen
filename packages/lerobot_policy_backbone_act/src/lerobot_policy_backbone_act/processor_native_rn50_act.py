from lerobot.policies.act.processor_act import make_act_pre_post_processors

from .configuration_native_rn50_act import NativeRN50ACTConfig


def make_native_rn50_act_pre_post_processors(
    config: NativeRN50ACTConfig, dataset_stats=None
):
    """Use the unmodified official ACT state/action processing pipeline."""

    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
