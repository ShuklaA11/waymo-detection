"""CLI entry point for the Waymo detection pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Waymo Detection Pipeline — detect Waymo vehicles and extract clips",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--video",
        help="Process a single video file (overrides input_videos in config)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Limit processing to first N frames (for testing)",
    )
    parser.add_argument(
        "--detection-mode",
        choices=["two_stage", "finetuned"],
        help="Override detection mode",
    )
    parser.add_argument(
        "--model",
        help="Override model weights path",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        help="Override compute device",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory",
    )

    args = parser.parse_args()

    # Build config overrides from CLI args
    overrides = {}
    if args.video:
        overrides["input_videos"] = [args.video]
    if args.max_frames:
        overrides["max_frames"] = args.max_frames
    if args.detection_mode:
        overrides["detection_mode"] = args.detection_mode
    if args.model:
        overrides["model_weights"] = args.model
    if args.device:
        overrides["device"] = args.device
    if args.output_dir:
        overrides["output_dir"] = args.output_dir

    # Load config
    config = load_config(args.config, **overrides)

    # Determine base directory (for resolving relative paths)
    base_dir = str(Path(args.config).parent.resolve())

    print("Waymo Detection Pipeline")
    print(f"  Config: {args.config}")
    print(f"  Mode: {config.detection_mode}")
    print(f"  Device: {config.device}")
    print(f"  Output: {config.output_dir}")
    print()

    run_pipeline(config, base_dir=base_dir)


if __name__ == "__main__":
    main()
