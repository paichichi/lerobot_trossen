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


class RecoveringCamera:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.max_ages: list[float] = []

    def read_latest(self, *, max_age_ms: float):
        self.max_ages.append(max_age_ms)
        if self.failures:
            self.failures -= 1
            raise TimeoutError("capture thread stopped")
        return "fresh-frame"

    def connect(self) -> None:
        self.connect_calls += 1

    def disconnect(self) -> None:
        self.disconnect_calls += 1


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


def test_camera_drop_restarts_only_failed_camera(monkeypatch) -> None:
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
    camera = RecoveringCamera(failures=1)
    robot = SimpleNamespace(
        config=SimpleNamespace(
            camera_frame_max_age_ms=500.0,
            camera_reconnect_attempts=3,
            camera_reconnect_delay_s=0.0,
        )
    )

    frame = follower_module.WidowXAIFollower._read_camera_with_recovery(
        robot,
        "cam_main",
        camera,
        depth=False,
    )

    assert frame == "fresh-frame"
    assert camera.disconnect_calls == 1
    assert camera.connect_calls == 1
    assert camera.max_ages == [500.0, 500.0]


def test_camera_drop_fails_closed_after_retries(monkeypatch) -> None:
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
    camera = RecoveringCamera(failures=10)
    robot = SimpleNamespace(
        config=SimpleNamespace(
            camera_frame_max_age_ms=500.0,
            camera_reconnect_attempts=2,
            camera_reconnect_delay_s=0.0,
        )
    )

    try:
        follower_module.WidowXAIFollower._read_camera_with_recovery(
            robot,
            "cam_wrist",
            camera,
            depth=False,
        )
    except RuntimeError as error:
        assert "could not recover after 2 attempt(s)" in str(error)
    else:
        raise AssertionError("camera recovery should fail closed")

    assert camera.disconnect_calls == 2
    assert camera.connect_calls == 2
