import sys
from threading import Event, Lock
from unittest.mock import Mock

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

# The camera recovery tests do not construct an arm driver. The macOS test
# environment cannot install Trossen's Linux-only binary package.
sys.modules.setdefault("trossen_arm", Mock())
sys.modules.setdefault("trossen_slate", Mock())

from lerobot_robot_trossen.recovering_realsense import RecoveringRealSenseCamera


def _camera_without_hardware() -> RecoveringRealSenseCamera:
    camera = object.__new__(RecoveringRealSenseCamera)
    camera.serial_number = "camera-1"
    camera.rs_pipeline = None
    camera.rs_profile = None
    camera.thread = None
    camera.stop_event = None
    camera.frame_lock = Lock()
    camera.latest_color_frame = None
    camera.latest_depth_frame = None
    camera.latest_timestamp = None
    camera.new_frame_event = Event()
    return camera


def test_connect_retries_and_resets_before_final_attempt(monkeypatch) -> None:
    camera = _camera_without_hardware()
    attempts = 0
    reset = Mock(return_value=True)

    def connect_once(_camera, warmup=True):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError("no frames")
        _camera.rs_pipeline = Mock()
        _camera.rs_profile = Mock()

    monkeypatch.setattr(RealSenseCamera, "connect", connect_once)
    monkeypatch.setattr(camera, "_hardware_reset", reset)

    camera.connect()

    assert attempts == 3
    reset.assert_called_once_with()
    assert camera.is_connected


def test_failed_attempt_stops_partial_pipeline(monkeypatch) -> None:
    camera = _camera_without_hardware()
    failed_pipeline = Mock()
    attempts = 0

    def connect_once(_camera, warmup=True):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            _camera.rs_pipeline = failed_pipeline
            _camera.rs_profile = Mock()
            raise TimeoutError("warmup timed out")
        _camera.rs_pipeline = Mock()
        _camera.rs_profile = Mock()

    monkeypatch.setattr(RealSenseCamera, "connect", connect_once)

    camera.connect()

    failed_pipeline.stop.assert_called_once_with()
    assert camera.is_connected


def test_disconnect_stops_pipeline_and_clears_connection_state() -> None:
    camera = _camera_without_hardware()
    pipeline = Mock()
    camera.rs_pipeline = pipeline
    camera.rs_profile = Mock()

    camera.disconnect()

    pipeline.stop.assert_called_once_with()
    assert not camera.is_connected
    assert camera.thread is None
    assert camera.stop_event is None
