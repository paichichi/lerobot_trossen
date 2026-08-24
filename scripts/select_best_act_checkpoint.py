"""Select the saved checkpoint with the lowest held-out ACT validation loss."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVAL_PATTERN = re.compile(r"step\s+(\d+):\s+eval_loss=([0-9.eE+-]+)")


def find_best_checkpoint(
    output_dir: Path, log_path: Path | None = None
) -> tuple[int, float, Path]:
    log_path = log_path or output_dir / "train.log"
    matches = [
        (int(step), float(loss))
        for step, loss in EVAL_PATTERN.findall(log_path.read_text())
    ]
    if not matches:
        raise RuntimeError(f"No validation losses found in {log_path}")

    step, loss = min(matches, key=lambda item: item[1])
    checkpoint = output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Best validation step {step} has no saved checkpoint at {checkpoint}; "
            "eval_steps and save_freq must match"
        )
    return step, loss, checkpoint.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()

    step, loss, checkpoint = find_best_checkpoint(args.output_dir, args.log_path)
    result = {"step": step, "eval_loss": loss, "checkpoint": str(checkpoint)}
    (args.output_dir / "best_checkpoint.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output_dir / "best_checkpoint.txt").write_text(f"{checkpoint}\n")
    print(checkpoint)


if __name__ == "__main__":
    main()
