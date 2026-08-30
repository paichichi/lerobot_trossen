#!/usr/bin/env python3
"""Reset the C920 capture stream, then lock focus and exposure via UVC."""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import struct
import sys
import time

import cv2


DEFAULT_DEVICE = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
DEFAULT_FOCUS = 19
DEFAULT_EXPOSURE = 39
DEFAULT_WHITE_BALANCE = 3994
DEFAULT_RESET_FRAMES = 5
DEFAULT_RESET_ATTEMPTS = 3

VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C

CONTROL_IDS = {
    "autofocus": 0x009A090C,
    "focus": 0x009A090A,
    "auto_exposure": 0x009A0901,
    "exposure": 0x009A0902,
    "auto_white_balance": 0x0098090C,
    "white_balance_temperature": 0x0098091A,
    "zoom": 0x009A090D,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--focus", type=int, default=DEFAULT_FOCUS)
    parser.add_argument("--exposure", type=int, default=DEFAULT_EXPOSURE)
    parser.add_argument("--white-balance", type=int, default=DEFAULT_WHITE_BALANCE)
    parser.add_argument("--reset-frames", type=int, default=DEFAULT_RESET_FRAMES)
    parser.add_argument("--reset-attempts", type=int, default=DEFAULT_RESET_ATTEMPTS)
    return parser.parse_args()


def set_control(fd: int, control_id: int, value: int) -> None:
    fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("Ii", control_id, value))


def get_control(fd: int, control_id: int) -> int:
    result = fcntl.ioctl(fd, VIDIOC_G_CTRL, struct.pack("Ii", control_id, 0))
    return struct.unpack("Ii", result)[1]


def reset_capture_stream(device: str, frame_count: int, attempts: int) -> bool:
    """Open, validate, and release the same C920 stream used by rollout."""
    device_node = os.path.realpath(device)
    match = re.fullmatch(r"/dev/video([0-9]+)", device_node)
    if match is None:
        print(f"C920 path did not resolve to /dev/videoN: {device} -> {device_node}", file=sys.stderr)
        return False
    camera_index = int(match.group(1))

    for attempt in range(1, attempts + 1):
        # This OpenCV build accepts a numeric V4L2 index, but not a by-id path
        # string when CAP_V4L2 is selected explicitly.
        capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        reset_ok = False
        try:
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                capture.set(cv2.CAP_PROP_FPS, 20)
                reset_ok = True
                for frame_index in range(frame_count):
                    ok, frame = capture.read()
                    if not ok or frame is None or frame.shape[:2] != (480, 640):
                        reset_ok = False
                        break
        finally:
            capture.release()

        if reset_ok:
            # Give the driver time to release the stream before rollout reopens it.
            time.sleep(0.25)
            print(
                f"C920 capture reset: {frame_count} MJPG frames at 640x480, "
                f"20 FPS via {device_node} (attempt {attempt}/{attempts})"
            )
            return True
        print(
            f"C920 reset attempt {attempt}/{attempts} failed at {device_node}",
            file=sys.stderr,
        )
        time.sleep(0.5)

    return False


def main() -> int:
    args = parse_args()
    if args.reset_frames < 1 or args.reset_attempts < 1:
        print("--reset-frames and --reset-attempts must be at least 1", file=sys.stderr)
        return 2
    if not reset_capture_stream(args.device, args.reset_frames, args.reset_attempts):
        print(f"Could not reset the C920 video stream at {args.device}", file=sys.stderr)
        return 1

    requested = {
        "autofocus": 0,
        "focus": args.focus,
        "auto_exposure": 1,
        "exposure": args.exposure,
        "auto_white_balance": 0,
        "white_balance_temperature": args.white_balance,
        "zoom": 100,
    }

    try:
        fd = os.open(args.device, os.O_RDWR | os.O_NONBLOCK)
    except OSError as error:
        print(f"Could not open C920 controls at {args.device}: {error}", file=sys.stderr)
        return 1
    try:
        for name, value in requested.items():
            set_control(fd, CONTROL_IDS[name], value)
        actual = {name: get_control(fd, CONTROL_IDS[name]) for name in requested}
    except OSError as error:
        print(f"C920 rejected a UVC control: {error}", file=sys.stderr)
        return 1
    finally:
        os.close(fd)

    mismatches = {
        name: (requested[name], actual[name])
        for name in requested
        if requested[name] != actual[name]
    }
    print("C920 locked controls: " + ", ".join(f"{k}={v}" for k, v in actual.items()))
    if mismatches:
        print(f"C920 control verification failed: {mismatches}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
