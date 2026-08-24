"""Create ACT's seeded random episode order for an 80/20 train/validation split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata


def shuffled_episode_order(
    episode_tasks: list[list[str]], seed: int
) -> tuple[list[int], dict[str, tuple[int, int]]]:
    """Shuffle episodes independently per task and report the resulting 80/20 counts."""
    task_to_episodes: dict[str, list[int]] = {}
    for episode_index, tasks in enumerate(episode_tasks):
        task = tasks[0] if tasks else ""
        task_to_episodes.setdefault(task, []).append(episode_index)

    rng = random.Random(seed)
    order: list[int] = []
    counts: dict[str, tuple[int, int]] = {}
    for task, episodes in task_to_episodes.items():
        rng.shuffle(episodes)
        n_validation = (len(episodes) + 4) // 5
        counts[task] = (len(episodes) - n_validation, n_validation)
        order.extend(episodes)
    return order, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    episode_tasks = metadata.episodes["tasks"]
    order, counts = shuffled_episode_order(episode_tasks, args.seed)
    for task, (n_train, n_validation) in counts.items():
        print(
            f"{task or '<empty task>'}: {n_train} train, "
            f"{n_validation} validation (seed={args.seed})",
            file=sys.stderr,
        )
    print(json.dumps(order, separators=(",", ":")))


if __name__ == "__main__":
    main()
