import pytest
from lerobot.configs import FeatureType, PolicyFeature
from lerobot_policy_backbone_act.configuration_layerwise_act import (
    ACTRN50LayerwiseConfig,
)
from lerobot_policy_backbone_act.modeling_layerwise_act import (
    ACTRN50LayerwisePolicy,
)


def make_config() -> ACTRN50LayerwiseConfig:
    return ACTRN50LayerwiseConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.images.cam_main": PolicyFeature(
                FeatureType.VISUAL, (3, 64, 64)
            ),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
        pretrained_backbone_weights=None,
        device="cpu",
        push_to_hub=False,
        chunk_size=4,
        n_action_steps=4,
    )


def test_layerwise_policy_partitions_every_trainable_parameter_once() -> None:
    policy = ACTRN50LayerwisePolicy(make_config())
    groups = policy.get_optim_params()

    assert [group["name"] for group in groups] == [
        "act_and_projection",
        "backbone_layer3_layer4",
        "backbone_stem_layer1_layer2",
    ]
    assert [group["lr"] for group in groups] == [1e-5, 3e-6, 3e-7]

    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = [
        id(parameter) for parameter in policy.parameters() if parameter.requires_grad
    ]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)


def test_layerwise_policy_assigns_rn50_stages_to_expected_groups() -> None:
    policy = ACTRN50LayerwisePolicy(make_config())
    groups = policy.get_optim_params()
    group_by_parameter = {
        id(parameter): group["name"]
        for group in groups
        for parameter in group["params"]
    }

    for name, parameter in policy.named_parameters():
        group_name = group_by_parameter[id(parameter)]
        if name.startswith(("model.backbone.layer3.", "model.backbone.layer4.")):
            assert group_name == "backbone_layer3_layer4"
        elif name.startswith(
            (
                "model.backbone.conv1.",
                "model.backbone.bn1.",
                "model.backbone.layer1.",
                "model.backbone.layer2.",
            )
        ):
            assert group_name == "backbone_stem_layer1_layer2"
        else:
            assert group_name == "act_and_projection"


def test_layerwise_scheduler_preserves_lr_ratios_and_decays_to_ten_percent() -> None:
    config = make_config()
    scheduler = config.get_scheduler_preset()

    assert scheduler.num_warmup_steps == 500
    assert scheduler.num_decay_steps == 8000
    assert scheduler.peak_lr == 1e-5
    assert scheduler.decay_lr == pytest.approx(1e-6)
