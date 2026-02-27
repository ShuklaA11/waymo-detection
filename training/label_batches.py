"""
Interactive batch labeling tool for Waymo vehicle crops.

Shows 20 images at a time in a grid. User selects which are Waymo vehicles.
Selected → moved to waymo/, rest → moved to not_waymo/.

Usage:
    python training/label_batches.py
    python training/label_batches.py --batch-size 20
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def make_grid(images: list[np.ndarray], labels: list[str], cols: int = 5) -> np.ndarray:
    """Create a labeled grid of images."""
    cell_w, cell_h = 200, 180
    label_h = 30
    rows = (len(images) + cols - 1) // cols
    grid_h = rows * (cell_h + label_h)
    grid_w = cols * cell_w
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark gray bg

    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        y_off = r * (cell_h + label_h)
        x_off = c * cell_w

        # Resize image to fit cell
        h, w = img.shape[:2]
        scale = min((cell_w - 10) / w, (cell_h - 10) / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Center in cell
        py = y_off + (cell_h - new_h) // 2
        px = x_off + (cell_w - new_w) // 2
        grid[py:py + new_h, px:px + new_w] = resized

        # Draw label
        label_y = y_off + cell_h + 20
        cv2.putText(grid, label, (x_off + 5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return grid


def main():
    parser = argparse.ArgumentParser(description="Batch label vehicle crops")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--source", default="training/dataset/unsorted")
    parser.add_argument("--waymo-dir", default="training/dataset/waymo")
    parser.add_argument("--not-waymo-dir", default="training/dataset/not_waymo")
    args = parser.parse_args()

    source = Path(args.source)
    waymo_dir = Path(args.waymo_dir)
    not_waymo_dir = Path(args.not_waymo_dir)
    waymo_dir.mkdir(parents=True, exist_ok=True)
    not_waymo_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(source.glob("*.jpg"))
    total = len(all_images)
    if total == 0:
        print("No images in unsorted/. Run filter_white_crops.py first.")
        return

    total_batches = (total + args.batch_size - 1) // args.batch_size
    print(f"Found {total} images to label ({total_batches} batches of {args.batch_size})")
    print(f"Controls:")
    print(f"  Type numbers of Waymo images (e.g. '1,5,13')")
    print(f"  Type 'none' or press Enter if no Waymos")
    print(f"  Type 'quit' to stop and resume later")
    print(f"  Type 'back' to undo last batch")
    print()

    waymo_total = 0
    batch_num = 0
    history = []  # for undo

    while True:
        # Re-read directory each batch (in case of undo)
        remaining = sorted(source.glob("*.jpg"))
        if not remaining:
            print(f"\nAll done! Labeled all images.")
            break

        batch = remaining[:args.batch_size]
        batch_num += 1
        batches_left = (len(remaining) + args.batch_size - 1) // args.batch_size

        print(f"--- Batch {batch_num} ({len(remaining)} images remaining, {batches_left} batches left) ---")

        # Load and display
        images = []
        labels = []
        for i, p in enumerate(batch):
            img = cv2.imread(str(p))
            if img is None:
                img = np.zeros((100, 100, 3), dtype=np.uint8)
            images.append(img)
            labels.append(f"[{i + 1}]")

        grid = make_grid(images, labels)
        cv2.imshow("Label Batch (close window or press any key to continue)", grid)
        cv2.waitKey(1)  # show window without blocking

        user_input = input("Waymo images (numbers, 'none', 'quit', 'back'): ").strip().lower()

        if user_input == "quit":
            print(f"\nStopped. {len(remaining)} images remaining in unsorted/.")
            print(f"Waymos found so far: {waymo_total}")
            break

        if user_input == "back" and history:
            last_batch, last_waymo_indices = history.pop()
            for p, dest in last_batch:
                # Move back to unsorted
                current = dest / p.name
                if current.exists():
                    shutil.move(str(current), str(source / p.name))
            batch_num -= 2  # will increment at top of loop
            waymo_total -= len(last_waymo_indices)
            print("  Undid last batch.")
            continue
        elif user_input == "back":
            print("  Nothing to undo.")
            continue

        # Parse waymo selections
        waymo_indices = set()
        if user_input and user_input != "none":
            for part in user_input.replace(" ", ",").split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    if 1 <= idx <= len(batch):
                        waymo_indices.add(idx - 1)

        # Move files
        batch_record = []
        for i, p in enumerate(batch):
            if i in waymo_indices:
                dest = waymo_dir
            else:
                dest = not_waymo_dir
            shutil.move(str(p), str(dest / p.name))
            batch_record.append((p, dest))

        history.append((batch_record, waymo_indices))
        waymo_total += len(waymo_indices)

        if waymo_indices:
            print(f"  Marked {len(waymo_indices)} as Waymo, {len(batch) - len(waymo_indices)} as not_waymo")
        else:
            print(f"  All {len(batch)} marked as not_waymo")

    cv2.destroyAllWindows()
    print(f"\nTotal Waymos found: {waymo_total}")
    print(f"waymo/ count: {len(list(waymo_dir.glob('*.jpg')))}")
    print(f"not_waymo/ count: {len(list(not_waymo_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
