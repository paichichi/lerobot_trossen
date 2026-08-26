from pathlib import Path

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot_policy_backbone_act.modeling_backbone_act import BackboneACTPolicy
from lerobot_policy_backbone_act.modeling_native_rn50_act import ACTRN50FullPolicy

REPO_ROOT = Path(__file__).parents[1]
EVAL_LAUNCHERS = (
    REPO_ROOT / "scripts/run_uploaded_act.sh",
    REPO_ROOT / "scripts/run_uploaded_v11.sh",
)

SHARED_DRIVER_ARGUMENTS = (
    "--robot.type=widowxai_follower_robot",
    "--robot.ip_address=192.168.1.4",
    "--robot.id=follower",
    "--robot.loop_rate=20",
    "--robot.min_time_to_move_multiplier=2.0",
    "--robot.max_relative_target=",
    "--robot.cameras=",
    "cam_main: {type: intelrealsense",
    'serial_number_or_name: "838212073584"',
    "cam_wrist: {type: intelrealsense",
    'serial_number_or_name: "409122274608"',
    "--strategy.type=base",
    "--fps=20",
    "--return_to_initial_position=true",
)

REMOVED_CUSTOM_DRIVER_ARGUMENTS = (
    "startup_home_",
    "controller_connect_",
    "camera_frame_max_age_",
    "camera_reconnect_",
    "fold_on_disconnect",
    "folded_positions",
    "arm_max_velocity_",
    "arm_max_acceleration_",
    "gripper_max_velocity_",
    "gripper_deadband_",
    "postprocess_",
    "hardware_rollout_",
)


def test_act_and_mlp_launchers_share_one_official_driver_contract() -> None:
    scripts = [path.read_text() for path in EVAL_LAUNCHERS]

    for argument in SHARED_DRIVER_ARGUMENTS:
        assert all(argument in script for script in scripts), argument
    for custom_argument in REMOVED_CUSTOM_DRIVER_ARGUMENTS:
        assert all(custom_argument not in script for script in scripts), custom_argument


def test_rn18_and_rn50_share_the_same_act_rollout_command() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()

    assert 'model="${1:-ours_rn50}"' in script
    assert "ours_rn50)" in script
    assert "rn18)" in script
    assert script.count("rollout_command=(") == 1


def test_rn50_full_alone_uses_a_lightweight_stateless_spike_cap() -> None:
    act_script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    v11_script = (REPO_ROOT / "scripts/run_uploaded_v11.sh").read_text()

    assert "max_relative_target=0.07" in act_script
    assert "max_relative_target=0.06" in act_script
    assert '--robot.max_relative_target="$max_relative_target"' in act_script
    assert "arm_max_velocity" not in act_script
    assert "max_relative_target=0.06" not in v11_script


def test_launchers_pin_end_to_end_rn50_checkpoints() -> None:
    act_script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    v11_script = (REPO_ROOT / "scripts/run_uploaded_v11.sh").read_text()

    assert "Chipaipai/act-ours-rn50-end-to-end-carrot-100" in act_script
    assert "5f0fd733e9098ba2e4c7143d44ada99087e0ae7d" in act_script
    assert "act-lite-ours-rn50-carrot-100" not in act_script
    assert "policies_v11_end_to_end_rn50/ours_rn50/checkpoint_100000.pt" in v11_script
    assert "56690ddea1023ebe840c2d0dd1a07cfad67377b0" in v11_script
    assert "policies_v11_basic_chunked_mlp" not in v11_script


def test_base_rollouts_do_not_create_evaluation_datasets() -> None:
    scripts = [path.read_text() for path in EVAL_LAUNCHERS]

    for script in scripts:
        assert "--strategy.type=base" in script
        assert "--dataset." not in script
        assert "rollout_dataset" not in script


def test_rn50_preserves_official_act_control_contract() -> None:
    assert issubclass(BackboneACTPolicy, ACTPolicy)
    assert issubclass(ACTRN50FullPolicy, ACTPolicy)
    action_methods = {
        "forward",
        "predict_action_chunk",
        "reset",
        "select_action",
        "update",
    }
    assert not action_methods.intersection(BackboneACTPolicy.__dict__)
    assert action_methods.intersection(ACTRN50FullPolicy.__dict__) == {
        "forward",
        "predict_action_chunk",
    }


def test_mlp_adapter_has_one_main_camera_and_absolute_joint_actions() -> None:
    export_source = (
        REPO_ROOT
        / "packages/lerobot_policy_v11/src/lerobot_policy_v11/export_v11.py"
    ).read_text()

    assert '"cameras": ["cam_main"]' in export_source
    assert '"action_representation": "absolute"' in export_source
    assert '"action_space": "joint_position"' in export_source
    assert '"observation.images.cam_main"' in export_source
    assert '"observation.state"' in export_source
