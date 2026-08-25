"""RealSense lifecycle recovery backported from current upstream LeRobot.

LeRobot 0.6.0 is retained for policy compatibility, but its RealSense camera
implementation predates upstream reconnect, hardware-reset, and teardown fixes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.cameras.realsense import camera_realsense
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError

logger = logging.getLogger(__name__)


class RecoveringRealSenseCamera(RealSenseCamera):
    """LeRobot 0.6.0 camera with upstream-style reconnect and teardown."""

    _MAX_CONNECT_ATTEMPTS = 3

    def _cleanup_resources(self) -> None:
        """Stop the read thread and pipeline without leaving a blocked USB reader."""
        read_thread = self.thread
        stop_event = self.stop_event
        rs_pipeline = self.rs_pipeline

        if stop_event is not None:
            stop_event.set()
        if read_thread is not None and read_thread.is_alive():
            read_thread.join(timeout=2.0)

        # Clear the public connection state before pipeline.stop(). Stopping the
        # pipeline unblocks a hardware read that may have outlived the first join.
        self.rs_pipeline = None
        self.rs_profile = None
        try:
            if rs_pipeline is not None:
                rs_pipeline.stop()
        finally:
            if read_thread is not None and read_thread.is_alive():
                read_thread.join(timeout=2.0)
                if read_thread.is_alive():  # pragma: no cover - hardware dependent
                    logger.warning(
                        "%s read thread remained alive after stopping the pipeline.",
                        self,
                    )

            self.thread = None
            self.stop_event = None
            with self.frame_lock:
                self.latest_color_frame = None
                self.latest_depth_frame = None
                self.latest_timestamp = None
                self.new_frame_event.clear()

    def _hardware_reset(self, wait_s: float = 5.0) -> bool:
        """Ask librealsense to re-enumerate this device like a USB replug."""
        rs = camera_realsense.rs
        if rs is None:  # pragma: no cover - constructor already guards this
            return False

        context = rs.context()
        for device in context.query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            if serial == self.serial_number:
                logger.warning("%s performing automatic hardware reset.", self)
                device.hardware_reset()
                time.sleep(wait_s)
                return True

        logger.warning("%s was not visible on USB; hardware reset skipped.", self)
        return False

    def connect(self, warmup: bool = True) -> None:
        """Connect, retrying failed startup and resetting USB before the last try."""
        if not warmup:
            super().connect(warmup=False)
            return

        last_error: Exception | None = None
        for attempt in range(1, self._MAX_CONNECT_ATTEMPTS + 1):
            if attempt == self._MAX_CONNECT_ATTEMPTS:
                self._hardware_reset()

            try:
                super().connect(warmup=True)
                return
            except (ConnectionError, TimeoutError) as error:
                last_error = error
                self._cleanup_resources()
                logger.warning(
                    "%s connection failed (attempt %d/%d): %s",
                    self,
                    attempt,
                    self._MAX_CONNECT_ATTEMPTS,
                    error,
                )

        raise ConnectionError(
            f"{self} failed to connect after {self._MAX_CONNECT_ATTEMPTS} attempts."
        ) from last_error

    def disconnect(self) -> None:
        """Release the camera with the corrected upstream teardown ordering."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )
        self._cleanup_resources()
        logger.info("%s disconnected.", self)


def make_trossen_cameras_from_configs(
    camera_configs: dict[str, Any],
) -> dict[str, Any]:
    """Use the recovery camera for RealSense and stock LeRobot for other types."""
    cameras: dict[str, Any] = {}
    for key, config in camera_configs.items():
        if config.type == "intelrealsense":
            cameras[key] = RecoveringRealSenseCamera(config)
        else:
            cameras.update(make_cameras_from_configs({key: config}))
    return cameras
