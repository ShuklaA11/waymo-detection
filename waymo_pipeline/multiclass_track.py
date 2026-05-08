"""Multi-class detection + ByteTrack tracking.

Reads a video, runs the fine-tuned 6-class YOLO with ByteTrack, and emits:
  - <out>/<stem>_tracks.csv  (one row per detection per frame)
  - <out>/<stem>_annotated.mp4  (boxes + class + track_id burned in)
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

CLASS_NAMES = ["AV", "BC", "HDV", "MC", "PED", "SCO"]

CLASS_COLORS = {
    "AV":  (0, 255, 0),
    "HDV": (0, 165, 255),
    "PED": (0, 0, 255),
    "BC":  (255, 0, 255),
    "MC":  (255, 255, 0),
    "SCO": (128, 0, 255),
}


def _draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def track_video(
    video_path: str,
    weights: str,
    out_dir: str,
    device: str = "0",
    conf: float = 0.25,
    tracker_yaml: str = "bytetrack.yaml",
):
    video_path = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{video_path.stem}_tracks.csv"
    mp4_path = out / f"{video_path.stem}_annotated.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))

    model = YOLO(weights)
    results = model.track(
        source=str(video_path),
        stream=True,
        tracker=tracker_yaml,
        conf=conf,
        device=device,
        persist=True,
        verbose=False,
    )

    f_csv = open(csv_path, "w", newline="")
    csv_w = csv.writer(f_csv)
    csv_w.writerow([
        "frame_num", "track_id", "class", "conf",
        "x1", "y1", "x2", "y2", "cx", "cy",
    ])

    frame_idx = 0
    for r in results:
        frame = r.orig_img.copy()
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                if b.id is None:
                    continue
                tid = int(b.id.item())
                cid = int(b.cls.item())
                cls = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
                conf_ = float(b.conf.item())
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                csv_w.writerow([
                    frame_idx, tid, cls, f"{conf_:.3f}",
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                    f"{cx:.1f}", f"{cy:.1f}",
                ])
                color = CLASS_COLORS.get(cls, (200, 200, 200))
                _draw_box(
                    frame, int(x1), int(y1), int(x2), int(y2),
                    f"{cls} #{tid} {conf_:.2f}", color,
                )
        writer.write(frame)
        frame_idx += 1

    f_csv.close()
    writer.release()
    print(f"  frames: {frame_idx}")
    print(f"  CSV:    {csv_path}")
    print(f"  MP4:    {mp4_path}")
