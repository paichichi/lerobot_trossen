import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "packages/lerobot_robot_trossen/src/lerobot_robot_trossen/arm_postprocess.py"
)
SPEC = importlib.util.spec_from_file_location("arm_postprocess_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TimeAwareArmTargetFilter = MODULE.TimeAwareArmTargetFilter


def make_filter() -> TimeAwareArmTargetFilter:
    return TimeAwareArmTargetFilter(
        arm_joint_names=("joint_0", "joint_1"),
        nominal_dt=0.05,
        max_dt_multiplier=2.0,
        max_velocity_rad_s=0.5,
        max_acceleration_rad_s2=3.0,
    )


def test_arm_filter_ramps_velocity_and_leaves_gripper_unchanged() -> None:
    target_filter = make_filter()
    present = {"joint_0": 0.0, "joint_1": 0.0, "left_carriage_joint": 0.0}
    desired = {"joint_0": 1.0, "joint_1": -1.0, "left_carriage_joint": 0.02}

    first = target_filter.apply(desired, present, now=10.0)
    second = target_filter.apply(desired, first, now=10.05)

    assert first["joint_0"] == pytest.approx(0.0075)
    assert first["joint_1"] == pytest.approx(-0.0075)
    assert second["joint_0"] - first["joint_0"] == pytest.approx(0.015)
    assert second["joint_1"] - first["joint_1"] == pytest.approx(-0.015)
    assert first["left_carriage_joint"] == pytest.approx(0.02)
    assert second["left_carriage_joint"] == pytest.approx(0.02)


def test_arm_filter_caps_slow_ticks_and_resets_after_a_stall() -> None:
    target_filter = make_filter()
    present = {"joint_0": 0.0, "joint_1": 0.0}
    desired = {"joint_0": 1.0, "joint_1": 0.0}

    target_filter.apply(desired, present, now=1.0)
    after_slow_tick = target_filter.apply(desired, present, now=1.2)
    after_stall = target_filter.apply(desired, present, now=2.0)

    assert after_slow_tick["joint_0"] == pytest.approx(0.045)
    assert after_stall["joint_0"] == pytest.approx(0.0075)
