"""Phase 4 step 1: pick ground-control points for the camera->map homography.

Run locally. Click ground landmarks on a reference frame (crosswalk corners,
stop bars, median posts — anything flat on the road plane and identifiable on
Google Maps). Close the window, then paste each point's lat/lng (right-click
the same spot in Google Maps to read it). Saves a GCP JSON + annotated image.

Pick >= 4 points (6-8 recommended), well spread across the road plane.

Usage:
    python pick_gcp.py --video "output/sjb/day/<midday_clip>.mp4" --camera sjb
    python pick_gcp.py --frame ref.png --camera sjb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt


def _load_frame(args):
    if args.frame:
        img = cv2.imread(args.frame)
        if img is None:
            raise SystemExit(f"Cannot read frame: {args.frame}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Cannot read frame from video: {args.video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--frame", default=None)
    ap.add_argument("--camera", required=True, help="camera id, e.g. sjb")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.video and not args.frame:
        raise SystemExit("Provide --video or --frame")

    out = Path(args.out) if args.out else Path("configs") / f"gcp_{args.camera}.json"
    annotated = out.with_suffix(".png")

    img = _load_frame(args)
    h, w = img.shape[:2]
    pts: list[tuple[float, float]] = []

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img)
    ax.set_title("Click ground landmarks (>=4). Right-click = undo. Close window when done.")

    def redraw():
        ax.clear()
        ax.imshow(img)
        ax.set_title("Click ground landmarks (>=4). Right-click = undo. Close window when done.")
        for i, (x, y) in enumerate(pts):
            ax.plot(x, y, "o", color="red", markersize=7)
            ax.text(x + 8, y - 8, str(i), color="yellow", fontsize=12, weight="bold")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:  # left
            pts.append((float(event.xdata), float(event.ydata)))
        elif event.button == 3 and pts:  # right = undo
            pts.pop()
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if len(pts) < 4:
        raise SystemExit(f"Need >= 4 points, got {len(pts)}. Nothing saved.")

    # Save annotated reference so the user can cross-check on Google Maps.
    fig2, ax2 = plt.subplots(figsize=(16, 9))
    ax2.imshow(img)
    for i, (x, y) in enumerate(pts):
        ax2.plot(x, y, "o", color="red", markersize=7)
        ax2.text(x + 8, y - 8, str(i), color="yellow", fontsize=12, weight="bold")
    ax2.axis("off")
    fig2.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(annotated, dpi=110)
    plt.close(fig2)
    print(f"Annotated reference saved: {annotated}")
    print(f"Open it alongside Google Maps and read each point's lat,lng.\n")

    # Collect lat/lng per point.
    records = []
    for i, (x, y) in enumerate(pts):
        raw = input(f"Point {i} @ pixel ({x:.0f},{y:.0f})  ->  lat,lng: ").strip()
        try:
            lat, lng = (float(v) for v in raw.replace(" ", "").split(","))
        except ValueError:
            print(f"  skipped point {i} (could not parse '{raw}')")
            continue
        records.append({"id": i, "px": [round(x, 1), round(y, 1)], "lat": lat, "lng": lng})

    if len(records) < 4:
        raise SystemExit(f"Need >= 4 points with coords, got {len(records)}. Nothing saved.")

    data = {"camera": args.camera, "frame_size": [w, h], "points": records}
    out.write_text(json.dumps(data, indent=2))
    print(f"\nSaved {len(records)} GCPs -> {out}")


if __name__ == "__main__":
    main()
