"""
Extract white vehicle crops from videos for classifier training.

Same as extract_crops.py but filters for white/light-colored vehicles only,
drastically reducing the number of images to manually classify.

Usage:
    python training/extract_white_crops.py --video data/video.mp4
    python training/extract_white_crops.py --all-videos
"""

from __future__ import annotations

import argparse
import glob
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


def is_white_vehicle(crop: np.ndarray, white_ratio_threshold: float = 0.25) -> bool:
    """Check if a vehicle crop is predominantly white/light colored.

    Uses HSV color space: white has low saturation and high value.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # White: low saturation, high value
    # Also catches light silver/grey which is fine — Waymo is white
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 60, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    ratio = np.count_nonzero(mask) / mask.size
    return ratio >= white_ratio_threshold


def process_video(
    video_path: str,
    output_dir: Path,
    model: YOLO,
    every: int = 5,
    max_frames: int | None = None,
    min_crop_size: int = 30,
    device: str = "mps",
    conf: float = 0.3,
    white_ratio: float = 0.25,
    video_prefix: str = "",
    process_every_n: int = 5,
) -> tuple[int, int]:
    """Process one video, return (saved_count, detection_count).

    process_every_n: only run YOLO on every Nth frame (skip others entirely).
    """
    vehicle_classes = [2, 5, 7]  # car, bus, truck

    print(f"\nProcessing: {video_path} (every {process_every_n}th frame)", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}", flush=True)
        return 0, 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total_frames}, will process ~{total_frames // process_every_n}", flush=True)

    detection_count = 0
    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if max_frames and frame_idx > max_frames:
            break

        # Skip frames for speed
        if frame_idx % process_every_n != 0:
            continue

        # Run YOLO on this frame
        results = model.predict(
            source=frame,
            classes=vehicle_classes,
            conf=conf,
            device=device,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            detection_count += 1

            if detection_count % every != 0:
                continue

            bbox = tuple(box.xyxy[0].tolist())
            crop = crop_bbox(frame, bbox)

            if crop is None:
                continue

            h, w = crop.shape[:2]
            if h < min_crop_size or w < min_crop_size:
                continue

            # White vehicle filter
            if not is_white_vehicle(crop, white_ratio):
                continue

            filename = f"{video_prefix}frame{frame_idx:06d}_det{detection_count:06d}.jpg"
            cv2.imwrite(str(output_dir / filename), crop)
            saved_count += 1

        if frame_idx % 2000 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%), white saved: {saved_count}", flush=True)

    cap.release()
    print(f"  Done: {frame_idx} frames, {saved_count} white crops saved", flush=True)
    return saved_count, detection_count


def main():
    parser = argparse.ArgumentParser(description="Extract white vehicle crops for classifier training")
    parser.add_argument("--video", help="Path to a single input video")
    parser.add_argument("--all-videos", action="store_true", help="Process all videos in data/")
    parser.add_argument("--output", default="training/dataset/strict_white_new", help="Output directory")
    parser.add_argument("--model", default="yolo26s.pt", help="YOLO26 model weights")
    parser.add_argument("--every", type=int, default=5, help="Save every Nth detection")
    parser.add_argument("--process-every-n", type=int, default=5, help="Only run YOLO on every Nth frame (speed vs coverage)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per video")
    parser.add_argument("--min-crop-size", type=int, default=30, help="Min crop dimension")
    parser.add_argument("--device", default="mps", help="Compute device")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence")
    parser.add_argument("--white-ratio", type=float, default=0.25, help="Min white pixel ratio")
    parser.add_argument("--skip-processed", default="data/EDK st and SJB-00-165621-174854.mp4",
                        help="Video to skip (already processed)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    if args.all_videos:
        videos = sorted(glob.glob("data/*.mp4"))
        # Skip the already-processed video
        if args.skip_processed:
            videos = [v for v in videos if args.skip_processed not in v]
        print(f"Found {len(videos)} videos to process")
    elif args.video:
        videos = [args.video]
    else:
        parser.error("Specify --video or --all-videos")

    total_saved = 0
    total_det = 0

    for i, video in enumerate(videos):
        # Use video index as prefix to avoid filename collisions
        prefix = f"v{i:02d}_"
        saved, det = process_video(
            video_path=video,
            output_dir=output_dir,
            model=model,
            every=args.every,
            max_frames=args.max_frames,
            min_crop_size=args.min_crop_size,
            device=args.device,
            conf=args.conf,
            white_ratio=args.white_ratio,
            video_prefix=prefix,
            process_every_n=args.process_every_n,
        )
        total_saved += saved
        total_det += det

    print(f"\n{'='*50}")
    print(f"All done! {len(videos)} videos processed")
    print(f"Total detections: {total_det}")
    print(f"Total white crops saved: {total_saved}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
