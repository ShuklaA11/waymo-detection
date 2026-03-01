"""
Extract cropped vehicle images from video for classifier training.

Runs pretrained YOLO26 on a video and saves every Nth detected vehicle crop
to the unsorted directory. User then manually sorts into waymo/ and not_waymo/.

Usage:
    python training/extract_crops.py --video data/video.mp4
    python training/extract_crops.py --video data/video.mp4 --every 10 --max-frames 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def crop_bbox(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    roof_extension: float = 0.2,
) -> np.ndarray | None:
    """Crop bounding box from frame, extending upward for roof."""
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    box_h = y2 - y1
    y1_ext = max(0, y1 - int(box_h * roof_extension))
    x1 = max(0, x1)
    x2 = min(w_frame, x2)
    y2 = min(h_frame, y2)
    crop = frame[y1_ext:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def main():
    parser = argparse.ArgumentParser(description="Extract vehicle crops for classifier training")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="training/dataset/unsorted", help="Output directory for crops")
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO26 model weights")
    parser.add_argument("--every", type=int, default=25, help="Save every Nth detection (default: 25)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--min-crop-size", type=int, default=30, help="Min crop dimension in pixels")
    parser.add_argument("--device", default="mps", help="Compute device")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    vehicle_classes = [2, 5, 7]  # car, bus, truck

    print(f"Processing: {args.video}")
    print(f"Saving crops to: {output_dir}")
    print(f"Saving every {args.every}th detection")

    results_gen = model.predict(
        source=args.video,
        stream=True,
        classes=vehicle_classes,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )

    detection_count = 0
    saved_count = 0
    frame_idx = 0

    for result in results_gen:
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            detection_count += 1

            # Only save every Nth detection
            if detection_count % args.every != 0:
                continue

            bbox = tuple(box.xyxy[0].tolist())
            crop = crop_bbox(result.orig_img, bbox)

            if crop is None:
                continue

            # Skip tiny crops
            h, w = crop.shape[:2]
            if h < args.min_crop_size or w < args.min_crop_size:
                continue

            # Save with descriptive filename
            filename = f"frame{frame_idx:06d}_det{detection_count:06d}.jpg"
            save_path = output_dir / filename
            cv2.imwrite(str(save_path), crop)
            saved_count += 1

        if frame_idx % 500 == 0:
            print(f"  Frame {frame_idx}, detections: {detection_count}, saved: {saved_count}")

    print(f"\nDone! Processed {frame_idx} frames")
    print(f"Total detections: {detection_count}")
    print(f"Saved crops: {saved_count}")
    print(f"\nNext steps:")
    print(f"  1. Open {output_dir}")
    print(f"  2. Move Waymo crops to training/dataset/waymo/")
    print(f"  3. Move non-Waymo crops to training/dataset/not_waymo/")
    print(f"  4. Run: python training/train_classifier.py")


if __name__ == "__main__":
    main()
