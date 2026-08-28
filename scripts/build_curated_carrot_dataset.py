#!/usr/bin/env python3
"""Build the selected carrot demos as a LeRobot v2.1 one-file-per-episode dataset.

The source demos are preserved. Unselected source demo directories are moved into
the source dataset's backup directory only after the curated dataset validates.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DATASETS = Path(__file__).resolve().parents[1] / "datasets"
SOURCE = DATASETS / "carrot_to_pot"
TARGET = DATASETS / "carrot_to_pot_40"
BACKUP = SOURCE / "backup"
SELECTED = [
    2, 3, 4, 6, 7, 19, 21, 22,
    24, 25, 32, 34, 37, 42, 43, 44,
    46, 47, 49, 54, 55, 56, 57, 58,
    59, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75,
]
ALL_IDS = list(range(1, 76))
UNSELECTED = sorted(set(ALL_IDS) - set(SELECTED))
TASK = "Pick up the carrot and place it in the pot."


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=4) + "\n")


def dump_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def compact_stats(stats: dict) -> dict:
    keys = ("min", "max", "mean", "std", "count")
    return {
        name: {key: value[key] for key in keys if key in value}
        for name, value in stats.items()
        if name in ("action", "observation.state", "observation.images.cam_main")
    }


def pooled_image_stats(episode_stats: list[dict], key: str) -> dict:
    parts = [np.asarray(s[key]["mean"], dtype=np.float64) for s in episode_stats]
    stds = [np.asarray(s[key]["std"], dtype=np.float64) for s in episode_stats]
    counts = [int(s[key]["count"][0]) for s in episode_stats]
    total = sum(counts)
    mean = sum(n * m for n, m in zip(counts, parts, strict=True)) / total
    second = sum(
        n * (sd**2 + m**2)
        for n, sd, m in zip(counts, stds, parts, strict=True)
    ) / total
    std = np.sqrt(np.maximum(0.0, second - mean**2))
    mins = [np.asarray(s[key]["min"], dtype=np.float64) for s in episode_stats]
    maxs = [np.asarray(s[key]["max"], dtype=np.float64) for s in episode_stats]
    return {
        "min": np.minimum.reduce(mins).tolist(),
        "max": np.maximum.reduce(maxs).tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [total],
    }


def build_info(source_info: dict) -> dict:
    features = deepcopy(source_info["features"])
    video_info = features["observation.images.cam_main"]["info"]
    features["observation.images.cam_main"]["info"] = {
        "video.fps": float(video_info["video.fps"]),
        "video.height": int(video_info["video.height"]),
        "video.width": int(video_info["video.width"]),
        "video.channels": int(video_info["video.channels"]),
        "video.codec": video_info["video.codec"],
        "video.pix_fmt": video_info["video.pix_fmt"],
        "video.is_depth_map": bool(video_info.get("is_depth_map", False)),
        "has_audio": bool(video_info.get("has_audio", False)),
    }
    return {
        "codebase_version": "v2.1",
        "trossen_subversion": "v1.0",
        "robot_type": source_info["robot_type"],
        "total_episodes": 40,
        "total_frames": 12000,
        "total_tasks": 1,
        "total_videos": 40,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 20,
        "splits": {"train": "0:40"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "repo_id": "local/carrot_to_pot_40",
    }


def validate_source() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Target already exists: {TARGET}")
    for demo_id in ALL_IDS:
        demo = SOURCE / f"demo_{demo_id:02d}"
        if not demo.is_dir():
            raise FileNotFoundError(f"Missing source directory: {demo}")
    conflicts = [BACKUP / f"demo_{i:02d}" for i in UNSELECTED if (BACKUP / f"demo_{i:02d}").exists()]
    if conflicts:
        raise FileExistsError(f"Backup destinations already exist: {conflicts[:3]}")


def validate_build(root: Path) -> None:
    data_files = sorted((root / "data/chunk-000").glob("episode_*.parquet"))
    video_files = sorted((root / "videos/chunk-000/observation.images.cam_main").glob("episode_*.mp4"))
    if len(data_files) != 40 or len(video_files) != 40:
        raise RuntimeError(f"Expected 40 parquet and 40 video files, got {len(data_files)} and {len(video_files)}")
    expected_global = 0
    for episode_index, (data_path, video_path) in enumerate(zip(data_files, video_files, strict=True)):
        table = pq.read_table(data_path)
        if table.num_rows != 300:
            raise RuntimeError(f"{data_path} has {table.num_rows} rows")
        frame_index = np.asarray(table["frame_index"].to_pylist())
        ep_index = np.asarray(table["episode_index"].to_pylist())
        global_index = np.asarray(table["index"].to_pylist())
        if not np.array_equal(frame_index, np.arange(300)):
            raise RuntimeError(f"Bad frame index in {data_path}")
        if not np.all(ep_index == episode_index):
            raise RuntimeError(f"Bad episode index in {data_path}")
        if not np.array_equal(global_index, np.arange(expected_global, expected_global + 300)):
            raise RuntimeError(f"Bad global index in {data_path}")
        expected_global += 300
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        decoded = sum(1 for _ in container.decode(stream))
        rate = float(stream.average_rate)
        width, height = stream.width, stream.height
        container.close()
        if (decoded, rate, width, height) != (300, 20.0, 640, 480):
            raise RuntimeError(
                f"Bad video {video_path}: frames={decoded}, fps={rate}, size={width}x{height}"
            )
    info = json.loads((root / "meta/info.json").read_text())
    if info["total_episodes"] != 40 or info["total_frames"] != 12000:
        raise RuntimeError("Invalid aggregate metadata")


def main() -> None:
    validate_source()
    temp_root = Path(tempfile.mkdtemp(prefix=".carrot_to_pot_40_build_", dir=DATASETS))
    data_dir = temp_root / "data/chunk-000"
    video_dir = temp_root / "videos/chunk-000/observation.images.cam_main"
    meta_dir = temp_root / "meta"
    data_dir.mkdir(parents=True)
    video_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)

    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    source_episode_stats: list[dict] = []
    episodes: list[dict] = []
    episodes_stats: list[dict] = []
    source_map: list[dict] = []
    source_info = None
    global_start = 0

    for episode_index, demo_id in enumerate(SELECTED):
        demo = SOURCE / f"demo_{demo_id:02d}"
        info = json.loads((demo / "meta/info.json").read_text())
        if source_info is None:
            source_info = info
        elif info["features"] != source_info["features"] or info["fps"] != source_info["fps"]:
            raise RuntimeError(f"Feature mismatch in demo_{demo_id:02d}")

        src_data = demo / "data/chunk-000/file-000.parquet"
        table = pq.read_table(src_data)
        if table.num_rows != 300:
            raise RuntimeError(f"demo_{demo_id:02d} has {table.num_rows} rows")
        n = table.num_rows
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array([episode_index] * n, type=pa.int64()),
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            pa.array(range(global_start, global_start + n), type=pa.int64()),
        )
        table = table.set_column(
            table.schema.get_field_index("task_index"),
            "task_index",
            pa.array([0] * n, type=pa.int64()),
        )
        dst_data = data_dir / f"episode_{episode_index:06d}.parquet"
        pq.write_table(table, dst_data, compression="zstd")

        src_video = demo / "videos/observation.images.cam_main/chunk-000/file-000.mp4"
        dst_video = video_dir / f"episode_{episode_index:06d}.mp4"
        shutil.copy2(src_video, dst_video)

        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        all_actions.append(action)
        all_states.append(state)
        episode_stats = compact_stats(json.loads((demo / "meta/stats.json").read_text()))
        source_episode_stats.append(episode_stats)
        episodes.append({"episode_index": episode_index, "tasks": [TASK], "length": n})
        episodes_stats.append({"episode_index": episode_index, "stats": episode_stats})
        source_map.append(
            {
                "episode_index": episode_index,
                "source_demo_id": demo_id,
                "source_directory": f"demo_{demo_id:02d}",
            }
        )
        global_start += n

    assert source_info is not None
    action = np.concatenate(all_actions)
    state = np.concatenate(all_states)
    aggregate_stats = {
        "observation.state": {
            "min": state.min(axis=0).tolist(),
            "max": state.max(axis=0).tolist(),
            "mean": state.mean(axis=0).tolist(),
            "std": state.std(axis=0).tolist(),
            "count": [len(state)],
        },
        "action": {
            "min": action.min(axis=0).tolist(),
            "max": action.max(axis=0).tolist(),
            "mean": action.mean(axis=0).tolist(),
            "std": action.std(axis=0).tolist(),
            "count": [len(action)],
        },
        "observation.images.cam_main": pooled_image_stats(
            source_episode_stats, "observation.images.cam_main"
        ),
    }
    info = build_info(source_info)
    dump_json(meta_dir / "info.json", info)
    dump_json(meta_dir / "stats.json", aggregate_stats)
    dump_jsonl(meta_dir / "episodes.jsonl", episodes)
    dump_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats)
    dump_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": TASK}])
    dump_jsonl(meta_dir / "episode_source_map.jsonl", source_map)
    (temp_root / "README.md").write_text(
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n- robotics\n"
        "tags:\n- LeRobot\n"
        "configs:\n- config_name: default\n  data_files: data/*/*.parquet\n"
        "---\n\n"
        "# Carrot to Pot — Curated 40\n\n"
        "A curated, single-camera WidowX AI dataset with 40 episodes, 300 frames per episode, "
        "and one parquet plus one MP4 per episode. See `meta/episode_source_map.jsonl` for the "
        "mapping back to the original `demo_XX` directories.\n"
    )

    validate_build(temp_root)
    temp_root.rename(TARGET)

    BACKUP.mkdir()
    for demo_id in UNSELECTED:
        (SOURCE / f"demo_{demo_id:02d}").rename(BACKUP / f"demo_{demo_id:02d}")
    (BACKUP / "README.txt").write_text(
        "Unselected source demos retained as recoverable backups.\n"
        f"IDs: {', '.join(f'{i:02d}' for i in UNSELECTED)}\n"
    )

    print(f"Created: {TARGET}")
    print(f"Curated episodes: {len(SELECTED)}, frames: {global_start}")
    print(f"Moved backup demos: {len(UNSELECTED)} -> {BACKUP}")


if __name__ == "__main__":
    main()
