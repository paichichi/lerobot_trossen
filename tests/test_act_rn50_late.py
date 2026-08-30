from lerobot.configs import FeatureType, PolicyFeature
from lerobot_policy_backbone_act.configuration_late_act import ACTRN50LateConfig
from lerobot_policy_backbone_act.modeling_late_act import ACTRN50LatePolicy


def make_config() -> ACTRN50LateConfig:
    return ACTRN50LateConfig(
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


def test_late_policy_freezes_only_early_rn50_stages() -> None:
    policy = ACTRN50LatePolicy(make_config())

    for name, parameter in policy.named_parameters():
        if name.startswith(policy._early_prefixes):
            assert not parameter.requires_grad, name
        elif name.startswith(policy._late_prefixes):
            assert parameter.requires_grad, name


def test_late_policy_partitions_every_trainable_parameter_once() -> None:
    policy = ACTRN50LatePolicy(make_config())
    groups = policy.get_optim_params()

    assert [group["name"] for group in groups] == [
        "act_and_projection",
        "backbone_layer3_layer4",
    ]
    assert [group["lr"] for group in groups] == [1e-5, 1e-6]

    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = [
        id(parameter) for parameter in policy.parameters() if parameter.requires_grad
    ]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)


def test_late_policy_keeps_the_previous_no_scheduler_baseline() -> None:
    assert make_config().get_scheduler_preset() is None
