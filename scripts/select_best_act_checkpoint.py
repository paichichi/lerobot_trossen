"""Select the saved checkpoint with the lowest held-out ACT validation loss."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EVAL_PATTERN = re.compile(r"step\s+(\d+):\s+eval_loss=([0-9.eE+-]+)")


def find_best_checkpoint(
    output_dir: Path,
    log_path: Path | None = None,
    *,
    image_swap_dir: Path | None = None,
    min_paired_ratio: float = 0.0,
    min_main_ratio: float = 0.0,
) -> tuple[int, float, Path]:
    log_path = log_path or output_dir / "train.log"
    matches = [
        (int(step), float(loss))
        for step, loss in EVAL_PATTERN.findall(log_path.read_text())
    ]
    if not matches:
        raise RuntimeError(f"No validation losses found in {log_path}")

    if image_swap_dir is not None:
        visually_grounded = []
        for step, loss in matches:
            report_path = image_swap_dir / f"{step:06d}.json"
            if not report_path.is_file():
                continue
            report = json.loads(report_path.read_text())
            predicted = report["predicted_action_dispersion"]
            paired_key = (
                "paired_images"
                if "paired_images" in predicted
                else "all_images"
            )
            paired_ratio = float(
                predicted[paired_key]["dispersion_ratio_vs_recorded"]
            )
            main_ratio = float(
                predicted["main_only"]["dispersion_ratio_vs_recorded"]
            )
            if paired_ratio >= min_paired_ratio and main_ratio >= min_main_ratio:
                visually_grounded.append((step, loss))
        if not visually_grounded:
            raise RuntimeError(
                "No saved checkpoint passed the visual-grounding gates: "
                f"paired>={min_paired_ratio}, main>={min_main_ratio}"
            )
        matches = visually_grounded

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
    parser.add_argument("--image-swap-dir", type=Path)
    parser.add_argument("--min-paired-ratio", type=float, default=0.0)
    parser.add_argument("--min-main-ratio", type=float, default=0.0)
    args = parser.parse_args()

    step, loss, checkpoint = find_best_checkpoint(
        args.output_dir,
        args.log_path,
        image_swap_dir=args.image_swap_dir,
        min_paired_ratio=args.min_paired_ratio,
        min_main_ratio=args.min_main_ratio,
    )
    result = {"step": step, "eval_loss": loss, "checkpoint": str(checkpoint)}
    if args.image_swap_dir is not None:
        report = json.loads((args.image_swap_dir / f"{step:06d}.json").read_text())
        predicted = report["predicted_action_dispersion"]
        paired_key = "paired_images" if "paired_images" in predicted else "all_images"
        result["paired_image_swap_ratio"] = predicted[paired_key][
            "dispersion_ratio_vs_recorded"
        ]
        result["main_image_swap_ratio"] = predicted["main_only"][
            "dispersion_ratio_vs_recorded"
        ]
    (args.output_dir / "best_checkpoint.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output_dir / "best_checkpoint.txt").write_text(f"{checkpoint}\n")
    print(checkpoint)


if __name__ == "__main__":
    main()
