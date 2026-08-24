import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


class FakeDriver:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.positions = [0.2] * 7

    def set_all_modes(self, mode: object) -> None:
        self.calls.append(("mode", mode))

    def set_all_positions(
        self, positions: list[float], *, goal_time: float, blocking: bool
    ) -> None:
        self.positions = list(positions)
        self.calls.append(("positions", list(positions), goal_time, blocking))

    def get_all_positions(self) -> list[float]:
        self.calls.append(("read",))
        return list(self.positions)

    def cleanup(self) -> None:
        self.calls.append(("cleanup",))


class FakeCamera:
    is_connected = True

    def __init__(self) -> None:
        self.disconnected = False

    def disconnect(self) -> None:
        self.disconnected = True
        self.is_connected = False


def test_safe_shutdown_folds_before_cleanup(monkeypatch) -> None:
    package_name = "lerobot_robot_trossen"
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(
            Path(__file__).parents[1]
            / "packages/lerobot_robot_trossen/src/lerobot_robot_trossen"
        )
    ]
    monkeypatch.setitem(sys.modules, package_name, package)

    fake_trossen = types.ModuleType("trossen_arm")
    fake_trossen.Mode = SimpleNamespace(position="position")
    monkeypatch.setitem(sys.modules, "trossen_arm", fake_trossen)

    follower_module = importlib.import_module(
        "lerobot_robot_trossen.widowxai_follower"
    )
    driver = FakeDriver()
    camera = FakeCamera()
    folded = [0.0] * 7
    robot = SimpleNamespace(
        driver=driver,
        cameras={"main": camera},
        _driver_configured=True,
        config=SimpleNamespace(
            fold_staging_goal_time_s=3.0,
            staged_positions=[0.0, 1.0, 0.5, 0.6, 0.0, 0.0, 0.0],
            fold_goal_time_s=3.0,
            folded_positions=folded,
            fold_max_arm_error_rad=0.12,
            fold_max_gripper_error_m=0.003,
        ),
    )

    follower_module.WidowXAIFollower._shutdown_hardware(robot, fold=True)

    position_calls = [call for call in driver.calls if call[0] == "positions"]
    assert position_calls[0][1] == robot.config.staged_positions
    assert position_calls[1][1] == folded
    assert driver.calls[-1] == ("cleanup",)
    assert robot._driver_configured is False
    assert camera.disconnected is True
