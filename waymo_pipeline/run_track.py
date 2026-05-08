"""CLI for multi-class detection + tracking.

Usage:
    python -m waymo_pipeline.run_track --video data/x.mp4 --weights runs/detect/waymo_6class/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .multiclass_track import track_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument(
        "--weights",
        default="runs/detect/waymo_6class/weights/best.pt",
        help="Fine-tuned YOLO weights",
    )
    ap.add_argument("--out", default="output/tracking", help="Output directory")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(
            f"Weights not found: {args.weights}\n"
            f"Train first with: python training/train_yolo_multiclass.py"
        )

    print(f"Tracking {args.video}")
    track_video(
        video_path=args.video,
        weights=args.weights,
        out_dir=args.out,
        device=args.device,
        conf=args.conf,
        tracker_yaml=args.tracker,
    )


if __name__ == "__main__":
    main()
