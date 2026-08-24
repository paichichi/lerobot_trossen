import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "packages/lerobot_robot_trossen/src/lerobot_robot_trossen/stable_postprocess.py"
)
SPEC = importlib.util.spec_from_file_location("stable_postprocess_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TimeAwareJointTargetFilter = MODULE.TimeAwareJointTargetFilter


def make_filter() -> TimeAwareJointTargetFilter:
    return TimeAwareJointTargetFilter(
        arm_joint_names=("joint_0", "joint_1"),
        gripper_joint_name="left_carriage_joint",
        nominal_dt=0.05,
        max_dt_multiplier=2.0,
        arm_max_velocity_rad_s=0.35,
        arm_max_acceleration_rad_s2=1.75,
        gripper_max_velocity_m_s=0.02,
        gripper_deadband_m=0.001,
    )


def test_stable_filter_ramps_arm_velocity_and_limits_gripper() -> None:
    target_filter = make_filter()
    present = {"joint_0": 0.0, "joint_1": 0.0, "left_carriage_joint": 0.0}
    desired = {"joint_0": 1.0, "joint_1": -1.0, "left_carriage_joint": 0.02}

    first = target_filter.apply(desired, present, now=10.0)
    second_present = dict(first)
    second = target_filter.apply(desired, second_present, now=10.05)

    assert first["joint_0"] == pytest.approx(0.004375)
    assert first["joint_1"] == pytest.approx(-0.004375)
    assert first["left_carriage_joint"] == pytest.approx(0.001)
    assert second["joint_0"] - second_present["joint_0"] == pytest.approx(0.00875)
    assert second["joint_1"] - second_present["joint_1"] == pytest.approx(-0.00875)
    assert second["left_carriage_joint"] - second_present["left_carriage_joint"] == pytest.approx(
        0.001
    )


def test_stable_filter_uses_actual_dt_but_caps_long_stalls() -> None:
    target_filter = make_filter()
    present = {"joint_0": 0.0, "joint_1": 0.0, "left_carriage_joint": 0.0}
    desired = {"joint_0": 1.0, "joint_1": 0.0, "left_carriage_joint": 0.0}

    target_filter.apply(desired, present, now=1.0)
    after_slow_tick = target_filter.apply(desired, present, now=1.08)
    after_long_stall = target_filter.apply(desired, present, now=2.0)

    assert after_slow_tick["joint_0"] == pytest.approx(0.0182)
    assert after_long_stall["joint_0"] == pytest.approx(0.004375)


def test_stable_filter_holds_gripper_inside_deadband() -> None:
    target_filter = make_filter()
    present = {"joint_0": 0.0, "joint_1": 0.0, "left_carriage_joint": 0.01}
    desired = {"joint_0": 0.0, "joint_1": 0.0, "left_carriage_joint": 0.0105}

    filtered = target_filter.apply(desired, present, now=1.0)

    assert filtered["left_carriage_joint"] == pytest.approx(0.01)
