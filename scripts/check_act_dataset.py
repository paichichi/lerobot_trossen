#!/usr/bin/env python3
"""Fail fast when a local dataset cannot be used by the ACT-lite recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_FPS = 20
EXPECTED_ACTION_DIM = 7
SUPPORTED_VERSIONS = {"v2.1", "v3.0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Dataset directory containing meta/info.json, data/, and videos/.",
    )
    parser.add_argument(
        "--require-v3",
        action="store_true",
        help="Also fail when the dataset still needs the official v2.1 to v3.0 conversion.",
    )
    return parser.parse_args()


def feature_shape(info: dict, key: str) -> list[int] | None:
    feature = info.get("features", {}).get(key)
    return feature.get("shape") if isinstance(feature, dict) else None


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"missing dataset metadata: {info_path}")

    info = json.loads(info_path.read_text())
    version = str(info.get("codebase_version", "unknown"))
    errors: list[str] = []

    if version not in SUPPORTED_VERSIONS:
        errors.append(f"unsupported codebase_version={version!r}")
    if args.require_v3 and version != "v3.0":
        errors.append("dataset must be converted to LeRobot v3.0 before training")
    if info.get("fps") != EXPECTED_FPS:
        errors.append(f"expected fps={EXPECTED_FPS}, got {info.get('fps')!r}")
    if feature_shape(info, "action") != [EXPECTED_ACTION_DIM]:
        errors.append(f"expected 7-D action, got {feature_shape(info, 'action')!r}")

    image_features = {
        key: value
        for key, value in info.get("features", {}).items()
        if isinstance(value, dict) and value.get("dtype") in {"video", "image"}
    }
    missing_views = {
        "observation.images.cam_main",
        "observation.images.cam_wrist",
    } - set(image_features)
    if missing_views:
        errors.append(f"missing required RGB views: {sorted(missing_views)}")

    shapes = {tuple(value.get("shape", [])) for value in image_features.values()}
    if len(shapes) > 1:
        errors.append(f"ACT requires equal image shapes; found {sorted(shapes)}")

    for directory in (root / "data", root / "videos"):
        if not directory.is_dir():
            errors.append(f"missing directory: {directory}")

    if errors:
        raise SystemExit("ACT dataset check failed:\n- " + "\n- ".join(errors))

    print("ACT dataset check passed")
    print(f"root={root}")
    print(f"codebase_version={version}")
    print(f"fps={info['fps']}")
    print(f"action_shape={feature_shape(info, 'action')}")
    print(f"image_features={sorted(image_features)}")
    if version == "v2.1":
        print("next=run the official v2.1 -> v3.0 converter before training")


if __name__ == "__main__":
    main()
