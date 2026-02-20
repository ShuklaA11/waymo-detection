"""
Filter vehicle crops by color, keeping only white/light-colored vehicles.

Moves images from not_waymo/ to unsorted/ if they pass the white-color filter
(mean saturation < 70 and mean value > 120 in HSV space).

Usage:
    python training/filter_white_crops.py
"""

from pathlib import Path
import shutil

import cv2


def is_white_vehicle(img_path: str, max_sat: float = 70, min_val: float = 120) -> bool:
    img = cv2.imread(img_path)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = hsv[:, :, 1].mean()
    mean_val = hsv[:, :, 2].mean()
    return mean_sat < max_sat and mean_val > min_val


def main():
    not_waymo = Path("training/dataset/not_waymo")
    unsorted = Path("training/dataset/unsorted")
    unsorted.mkdir(parents=True, exist_ok=True)

    images = sorted(not_waymo.glob("*.jpg"))
    print(f"Scanning {len(images)} images in not_waymo/ ...")

    moved = 0
    for i, img_path in enumerate(images):
        if is_white_vehicle(str(img_path)):
            shutil.move(str(img_path), str(unsorted / img_path.name))
            moved += 1
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(images)}, moved {moved} so far")

    remaining = len(images) - moved
    print(f"\nDone! Moved {moved} white/light crops to unsorted/")
    print(f"Remaining in not_waymo/: {remaining}")


if __name__ == "__main__":
    main()
