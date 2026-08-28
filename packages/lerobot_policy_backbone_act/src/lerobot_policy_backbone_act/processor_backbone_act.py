from lerobot.policies.act.processor_act import make_act_pre_post_processors

from .configuration_backbone_act import BackboneACTConfig


def make_backbone_act_pre_post_processors(
    config: BackboneACTConfig, dataset_stats=None
):
    """Keep official ACT state/action processing; the adapter owns RGB normalization."""

    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
