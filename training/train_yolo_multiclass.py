"""Fine-tune YOLO on the 6-class connected-intersection dataset.

Run after training/prepare_dataset.py has produced training/dataset_yolo/data.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/dataset_yolo/data.yaml")
    ap.add_argument("--weights", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="waymo_6class")
    ap.add_argument("--project", default="runs/detect")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(
            f"Missing {args.data}. Run training/prepare_dataset.py first."
        )

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=15,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Best weights: {best.resolve()}")


if __name__ == "__main__":
    main()
