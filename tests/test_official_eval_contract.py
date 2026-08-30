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
    "--robot.max_relative_target=",
    "--robot.cameras=",
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


def test_c920_rn18_and_rn50_share_the_same_act_rollout_command() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()

    assert 'model="${1:-rn50_newcam_8k}"' in script
    assert "rn18_newcam_2k|rn18_newcam_4k|rn18_newcam_6k|rn18_newcam_8k)" in script
    assert (
        "rn50_newcam_2k|rn50_newcam_4k|rn50_newcam_6k|rn50_newcam_8k|"
        "rn50_newcam_9k|rn50_newcam_10k)"
    ) in script
    assert script.count("rollout_command=(") == 1


def test_act_launcher_supports_an_explicit_replanning_override() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()

    assert 'n_action_steps_override="${3:-}"' in script
    assert 'rollout_command+=(--policy.n_action_steps="$n_action_steps_override")' in script
    assert 'n_action_steps must be an integer from 1 to the trained chunk_size of 40.' in script
    assert '${n_action_steps_override:-checkpoint_default}' in script


def test_act_launcher_only_creates_run_logs_for_execute_mode() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    execute_logging = script.index('if [[ "$mode" == --execute ]]; then')
    download = script.index("uv sync --extra act")

    assert execute_logging < script.index('mkdir -p "$run_dir"') < download
    assert execute_logging < script.index('tee -a "$run_dir/console.log"') < download
    assert 'run_dir=""' in script
    assert 'if [[ "$mode" == --execute ]]; then\n  sha256sum' in script


def test_new_camera_hf_checkpoints_are_all_selectable() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    for backbone in ("rn18", "rn50"):
        for step in ("2k", "4k", "6k", "8k"):
            model = f"{backbone}_newcam_{step}"
            assert model in script
            assert f"bash scripts/run_uploaded_act.sh {model} download" in commands
            assert f"bash scripts/run_uploaded_act.sh {model} --execute" in commands

    for step in ("9k", "10k"):
        model = f"rn50_newcam_{step}"
        assert model in script
        assert f"bash scripts/run_uploaded_act.sh {model} download" in commands
        assert f"bash scripts/run_uploaded_act.sh {model} --execute" in commands

    assert 'if [[ -f "$policy_path/model.safetensors" ]]' in script
    assert 'echo "Using local checkpoint: $policy_path"' in script

    assert "Chipaipai/act-official-rn18-carrot-to-pot-40-train40-8k" in script
    assert "Chipaipai/act-official-rn50-carrot-to-pot-40-train40-8k" in script
    assert 'policy_prefix="checkpoints/' in script


def test_spatial_rn50_8k_local_checkpoint_is_selectable() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    assert "rn50_spatial_8k" in script
    assert "act_official_rn50_carrot_to_pot_40_spatial_lr1e6_train40_8000steps" in script
    assert 'policy_path_override="$repo_root/outputs/train/' in script
    assert 'if [[ -n "$policy_path_override" ]]' in script
    assert "bash scripts/run_uploaded_act.sh rn50_spatial_8k download" in commands
    assert "bash scripts/run_uploaded_act.sh rn50_spatial_8k --execute" in commands


def test_color_rn50_8k_local_checkpoint_is_selectable() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    assert "rn50_color_2k|rn50_color_4k|rn50_color_6k|rn50_color_8k" in script
    assert "act_official_rn50_carrot_to_pot_40_color_lr1e6_train40_8000steps" in script
    assert "bash scripts/run_uploaded_act.sh rn50_color_6k download" in commands
    assert "bash scripts/run_uploaded_act.sh rn50_color_6k --execute" in commands
    assert "bash scripts/run_uploaded_act.sh rn50_color_8k download" in commands
    assert "bash scripts/run_uploaded_act.sh rn50_color_8k --execute" in commands


def test_color3_rn50_local_checkpoints_are_selectable() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    assert "rn50_color3_4k|rn50_color3_5k|rn50_color3_6k|rn50_color3_7k|rn50_color3_8k" in script
    assert "act_official_rn50_carrot_to_pot_40_color3_lr1e6_train40_8000steps" in script
    for step in range(4, 9):
        assert f"bash scripts/run_uploaded_act.sh rn50_color3_{step}k download" in commands
        assert f"bash scripts/run_uploaded_act.sh rn50_color3_{step}k --execute" in commands


def test_ours_vit_local_checkpoints_and_upstream_constructor_are_selectable() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    assert "vit_ours_2k|vit_ours_4k|vit_ours_6k|vit_ours_8k" in script
    assert "act_ours_vit_carrot_to_pot_40_color3_frozen_train40_8000steps" in script
    assert "assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt" in script
    assert "backbone_source_root_override=/home/robotarm/TCC-core" in script
    assert 'export BACKBONE_CHECKPOINT="$backbone_checkpoint_override"' in script
    assert 'export BACKBONE_SOURCE_ROOT="$backbone_source_root_override"' in script
    for step in range(2, 9, 2):
        assert f"bash scripts/run_uploaded_act.sh vit_ours_{step}k download" in commands
        assert f"bash scripts/run_uploaded_act.sh vit_ours_{step}k --execute" in commands


def test_frozen_backbone_comparison_checkpoints_are_selectable() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    variants = ("ours_vit_new", "ours_rn50_new", "d4r_imagenet", "hrp_imagenet")
    for variant in variants:
        assert f"{variant}_2k|{variant}_4k|{variant}_6k|{variant}_8k" in script
        assert f"bash scripts/run_uploaded_act.sh {variant}_2k download" in commands
        for step in range(2, 9, 2):
            assert f"bash scripts/run_uploaded_act.sh {variant}_{step}k --execute" in commands

    assert "act_ours_vit_frozen_carrot_to_pot_40_color3_train40_8000steps" in script
    assert "act_ours_rn50_frozen_carrot_to_pot_40_color3_train40_8000steps" in script
    assert "act_d4r_imagenet_frozen_carrot_to_pot_40_color3_train40_8000steps" in script
    assert "act_hrp_imagenet_frozen_carrot_to_pot_40_color3_train40_8000steps" in script


def test_e2e_vit_local_checkpoints_are_selectable_without_overwriting_frozen() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    assert "vit_e2e_2k|vit_e2e_4k|vit_e2e_6k|vit_e2e_8k" in script
    assert "act_ours_vit_carrot_to_pot_40_color3_e2e_lr1e7_train40_8000steps" in script
    assert "act_ours_vit_carrot_to_pot_40_color3_frozen_train40_8000steps" in script
    for step in range(2, 9, 2):
        assert f"bash scripts/run_uploaded_act.sh vit_e2e_{step}k download" in commands
        assert f"bash scripts/run_uploaded_act.sh vit_e2e_{step}k --execute" in commands

    assert "vit_last1_250|vit_last1_500|vit_last1_750|vit_last1_1k" in script
    assert "last1_lr1e7_from_frozen8k_1000steps" in script
    for step in ("250", "500", "750", "1k"):
        assert f"bash scripts/run_uploaded_act.sh vit_last1_{step} download" in commands
        assert f"bash scripts/run_uploaded_act.sh vit_last1_{step} --execute" in commands

    assert "vit_ln_1k|vit_ln_2k|vit_ln_3k|vit_ln_4k" in script
    assert "layernorm_only_lr1e6_train40_8000steps" in script
    for step in range(1, 9):
        assert f"bash scripts/run_uploaded_act.sh vit_ln_{step}k download" in commands
        assert f"bash scripts/run_uploaded_act.sh vit_ln_{step}k --execute" in commands

    assert "vit_later_1k|vit_later_2k|vit_later_3k|vit_later_4k" in script
    assert "later1_lr1e6_train40_8000steps" in script
    for step in range(1, 9):
        assert f"bash scripts/run_uploaded_act.sh vit_later_{step}k download" in commands
        assert f"bash scripts/run_uploaded_act.sh vit_later_{step}k --execute" in commands

    assert "vit_compact_1k|vit_compact_3k" in script
    assert "color3_compact_val20_4000steps" in script
    for step in ("1k", "3k"):
        assert f"bash scripts/run_uploaded_act.sh vit_compact_{step} download" in commands
        assert f"bash scripts/run_uploaded_act.sh vit_compact_{step} --execute" in commands


def test_layerwise_rn50_local_checkpoints_are_selectable_without_leaking_discovery_override() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    assert "rn50_layerwise_2k|rn50_layerwise_4k|rn50_layerwise_6k|rn50_layerwise_8k" in script
    assert "act_official_rn50_carrot_to_pot_40_color_layerwise_train40_8000steps" in script
    assert "--policy.discover_packages_path" not in script
    assert "bash scripts/run_uploaded_act.sh rn50_layerwise_8k download" in commands
    assert "bash scripts/run_uploaded_act.sh rn50_layerwise_8k --execute" in commands


def test_late_only_rn50_local_checkpoints_are_selectable() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    commands = (REPO_ROOT / "ACT_LITE_COMMANDS.txt").read_text()

    assert "rn50_late_2k|rn50_late_4k|rn50_late_6k|rn50_late_8k" in script
    assert "act_official_rn50_carrot_to_pot_40_color_late_train40_8000steps" in script
    assert "bash scripts/run_uploaded_act.sh rn50_late_8k download" in commands
    assert "bash scripts/run_uploaded_act.sh rn50_late_8k --execute" in commands


def test_new_camera_policies_use_only_the_collected_main_view() -> None:
    script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()

    assert "TROSSEN_C920_PATH" in script
    assert "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0" in script
    assert 'robot_cameras="{cam_main:' in script
    assert "type: opencv" in script
    assert "index_or_path:" in script
    assert "fps: 20" in script
    assert "fourcc: MJPG" in script
    assert "scripts/lock_c920_focus.py" in script
    assert 'c920_device_node="$(readlink -f "$c920_path")"' in script
    assert 'fuser "$c920_device_node"' in script
    assert "release_stale_c920_rollout" in script
    assert "release_stale_act_rollouts" in script
    assert "Disconnecting stale ACT rollout process group(s)" in script
    assert "C920 is held by stale rollout PID(s)" in script
    assert "C920 released; continuing with reset and reconnect." in script
    assert "refusing to stop it automatically" in script
    assert 'kill -KILL -- "-$holder_pgid"' in script
    assert "hardware_reset_c920" in script
    assert "/usr/bin/usbreset 046d:08e5" in script
    assert "pkexec" not in script
    assert "Resetting HD Pro Webcam C920 ... ok" in script
    assert "C920 re-enumerated at $c920_device_node" in script
    assert "camera_reset=usb_hardware_046d:08e5" in script

    reset_rule = (REPO_ROOT / "scripts/99-c920-usbreset.rules").read_text()
    assert 'ATTR{idVendor}=="046d"' in reset_rule
    assert 'ATTR{idProduct}=="08e5"' in reset_rule
    assert 'OWNER="robotarm"' in reset_rule
    assert '--robot.cameras="$robot_cameras"' in script

    focus_script = (REPO_ROOT / "scripts/lock_c920_focus.py").read_text()
    assert "reset_capture_stream" in focus_script
    assert 're.fullmatch(r"/dev/video([0-9]+)", device_node)' in focus_script
    assert "cv2.VideoCapture(camera_index, cv2.CAP_V4L2)" in focus_script
    assert 'cv2.VideoWriter_fourcc(*"MJPG")' in focus_script
    assert "C920 capture reset" in focus_script
    assert "DEFAULT_RESET_ATTEMPTS = 3" in focus_script
    for removed_camera_setting in (
        "intelrealsense",
        "RealSense",
        "cam_wrist",
        "serial_number_or_name",
        "838212073584",
        "409122274608",
    ):
        assert removed_camera_setting not in script


def test_c920_launcher_keeps_the_official_safety_cap() -> None:
    act_script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()

    assert "max_relative_target=0.07" in act_script
    assert "max_relative_target=0.06" not in act_script
    assert '--robot.max_relative_target="$max_relative_target"' in act_script
    assert "arm_max_velocity" not in act_script


def test_act_launcher_uses_official_raw_action_path() -> None:
    act_script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()
    follower = (
        REPO_ROOT
        / "packages/lerobot_robot_trossen/src/lerobot_robot_trossen/widowxai_follower.py"
    ).read_text()

    assert "action_ema_alpha" not in act_script
    assert "arm_action_deadband" not in act_script
    assert "--robot.min_time_to_move_multiplier" not in act_script
    assert "smooth_goal_positions" not in follower
    assert "ensure_safe_goal_position" in follower
    assert "self.driver.get_all_positions()" in follower


def test_act_launcher_does_not_offer_legacy_policy_variants() -> None:
    act_script = (REPO_ROOT / "scripts/run_uploaded_act.sh").read_text()

    for legacy_variant in (
        "rn50_full",
        "rn50_rms",
        "rn50_train100",
        "act-ours-rn50-end-to-end-carrot-100",
    ):
        assert legacy_variant not in act_script


def test_base_rollouts_do_not_create_evaluation_datasets() -> None:
    scripts = [path.read_text() for path in EVAL_LAUNCHERS]

    for script in scripts:
        assert "--strategy.type=base" in script
        assert "--dataset." not in script
        assert "rollout_dataset" not in script


def test_rn50_only_replaces_the_official_act_visual_input_path() -> None:
    policy_classes = (BackboneACTPolicy, ACTRN50FullPolicy)
    assert all(issubclass(policy_class, ACTPolicy) for policy_class in policy_classes)
    action_methods = {
        "forward",
        "predict_action_chunk",
        "reset",
        "select_action",
        "update",
    }
    assert all(
        not action_methods.intersection(policy_class.__dict__)
        for policy_class in policy_classes
    )


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
