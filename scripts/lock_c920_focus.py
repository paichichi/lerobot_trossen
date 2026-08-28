#!/usr/bin/env python3
"""Lock Logitech C920 focus, exposure, white balance, and zoom via UVC."""

from __future__ import annotations

import argparse
import fcntl
import os
import struct
import sys


DEFAULT_DEVICE = "/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0"
DEFAULT_FOCUS = 19
DEFAULT_EXPOSURE = 39
DEFAULT_WHITE_BALANCE = 3994

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
    return parser.parse_args()


def set_control(fd: int, control_id: int, value: int) -> None:
    fcntl.ioctl(fd, VIDIOC_S_CTRL, struct.pack("Ii", control_id, value))


def get_control(fd: int, control_id: int) -> int:
    result = fcntl.ioctl(fd, VIDIOC_G_CTRL, struct.pack("Ii", control_id, 0))
    return struct.unpack("Ii", result)[1]


def main() -> int:
    args = parse_args()
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
