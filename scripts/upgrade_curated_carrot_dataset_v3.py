#!/usr/bin/env python3
"""Make carrot_to_pot_40 loadable by LeRobot v3 while preserving per-episode files."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/carrot_to_pot"
TARGET = ROOT / "datasets/carrot_to_pot_40"
SELECTED = [
    2, 3, 4, 6, 7, 19, 21, 22,
    24, 25, 32, 34, 37, 42, 43, 44,
    46, 47, 49, 54, 55, 56, 57, 58,
    59, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75,
]


def lists_to_arrays(stats: dict) -> dict:
    return {
        feature: {name: np.asarray(value) for name, value in feature_stats.items()}
        for feature, feature_stats in stats.items()
    }


def arrays_to_lists(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: arrays_to_lists(item) for key, item in value.items()}
    return value


def adjusted_episode_stats(demo: Path, episode_index: int) -> dict:
    stats = deepcopy(json.loads((demo / "meta/stats.json").read_text()))
    global_start = episode_index * 300
    ep_stats = stats["episode_index"]
    for name in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        ep_stats[name] = [float(episode_index)]
    ep_stats["std"] = [0.0]
    index_stats = stats["index"]
    for name in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
        index_stats[name] = [float(index_stats[name][0] + global_start)]
    return stats


def main() -> None:
    meta = TARGET / "meta"
    current_info = json.loads((meta / "info.json").read_text())
    if current_info.get("codebase_version") == "v3.0":
        print(f"Already v3.0: {TARGET}")
        return

    legacy = meta / "legacy_v21"
    legacy.mkdir()
    shutil.copy2(meta / "info.json", legacy / "info.json")
    shutil.copy2(meta / "stats.json", legacy / "stats.json")
    for name in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        (meta / name).rename(legacy / name)

    source_info = json.loads((SOURCE / f"demo_{SELECTED[0]:02d}/meta/info.json").read_text())
    info = {
        "codebase_version": "v3.0",
        "fps": 20,
        "features": source_info["features"],
        "total_episodes": 40,
        "total_frames": 12000,
        "total_tasks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
        "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4",
        "robot_type": source_info["robot_type"],
        "splits": {"train": "0:40"},
    }

    episode_rows = []
    all_stats = []
    for episode_index, demo_id in enumerate(SELECTED):
        demo = SOURCE / f"demo_{demo_id:02d}"
        src_episode = pq.read_table(
            demo / "meta/episodes/chunk-000/file-000.parquet"
        ).to_pylist()[0]
        src_episode["episode_index"] = episode_index
        src_episode["data/chunk_index"] = 0
        src_episode["data/file_index"] = episode_index
        src_episode["dataset_from_index"] = episode_index * 300
        src_episode["dataset_to_index"] = (episode_index + 1) * 300
        video_prefix = "videos/observation.images.cam_main"
        src_episode[f"{video_prefix}/chunk_index"] = 0
        src_episode[f"{video_prefix}/file_index"] = episode_index
        src_episode[f"{video_prefix}/from_timestamp"] = 0.0
        src_episode[f"{video_prefix}/to_timestamp"] = 15.0
        src_episode["meta/episodes/chunk_index"] = 0
        src_episode["meta/episodes/file_index"] = 0

        episode_stats = adjusted_episode_stats(demo, episode_index)
        all_stats.append(lists_to_arrays(episode_stats))
        for feature, feature_stats in episode_stats.items():
            for stat_name, value in feature_stats.items():
                src_episode[f"stats/{feature}/{stat_name}"] = value
        episode_rows.append(src_episode)

    episodes_dir = meta / "episodes/chunk-000"
    episodes_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(episode_rows),
        episodes_dir / "file-000.parquet",
        compression="zstd",
    )
    shutil.copy2(
        SOURCE / f"demo_{SELECTED[0]:02d}/meta/tasks.parquet",
        meta / "tasks.parquet",
    )
    (meta / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    aggregate = arrays_to_lists(aggregate_stats(all_stats))
    (meta / "stats.json").write_text(json.dumps(aggregate, indent=4) + "\n")
    print(f"Upgraded metadata to v3.0: {TARGET}")


if __name__ == "__main__":
    main()
