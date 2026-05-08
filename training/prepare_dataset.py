"""Prepare YOLO-format dataset from labeled_data: random 80/20 split.

Reads images from <source>/image/*.png and matching YOLO-format labels from
<source>/class/*.txt. Writes copies into <out>/images/{train,val} and
<out>/labels/{train,val} alongside a data.yaml.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

CLASS_NAMES = ["AV", "BC", "HDV", "MC", "PED", "SCO"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default=r"D:\Video_recording\Connected_intersection\Labeled_data",
    )
    ap.add_argument("--out", default="training/dataset_yolo")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out)
    img_dir = src / "image"
    lbl_dir = src / "class"

    pairs = []
    for img in sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))

    if not pairs:
        raise SystemExit(f"No (image, label) pairs found under {src}")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = int(len(pairs) * args.val_frac)
    val, train = pairs[:n_val], pairs[n_val:]

    for split, items in [("train", train), ("val", val)]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(lbl, out / "labels" / split / lbl.name)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )

    print(f"train: {len(train)}  val: {len(val)}  -> {out.resolve()}")
    print(f"data.yaml: {data_yaml.resolve()}")


if __name__ == "__main__":
    main()
