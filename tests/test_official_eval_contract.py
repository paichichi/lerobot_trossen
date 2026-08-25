from pathlib import Path

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot_policy_backbone_act.modeling_backbone_act import BackboneACTPolicy

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
    "--robot.max_relative_target=0.07",
    "--robot.cameras=",
    "cam_main: {type: intelrealsense",
    'serial_number_or_name: "838212073584"',
    "cam_wrist: {type: intelrealsense",
    'serial_number_or_name: "409122274608"',
    "--strategy.type=episodic",
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
    assert script.count("record_command=(") == 1


def test_rn50_only_replaces_the_official_act_visual_input_path() -> None:
    assert issubclass(BackboneACTPolicy, ACTPolicy)
    action_methods = {
        "forward",
        "predict_action_chunk",
        "reset",
        "select_action",
        "update",
    }
    assert not action_methods.intersection(BackboneACTPolicy.__dict__)


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
