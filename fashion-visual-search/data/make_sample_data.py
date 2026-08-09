"""
Generates a tiny synthetic image dataset that mimics the structure of the
real subset (images/ + labels.csv with image_path, category, split), so the
entire pipeline (feature extraction -> transfer learning -> siamese ->
evaluation -> app) can be smoke-tested without internet access or a Kaggle
account.

Each "category" is rendered as a distinct shape/color combination so that a
CNN embedding genuinely can learn to tell them apart -- this is only for
pipeline verification, NOT a substitute for the real dataset.

Usage:
    python data/make_sample_data.py --per_category 30
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_DATA_DIR, SEED  # noqa: E402

CATEGORIES = {
    "Shirts":        {"color": (70, 130, 180), "shape": "rectangle"},
    "Tshirts":       {"color": (220, 20, 60),  "shape": "ellipse"},
    "Casual Shoes":  {"color": (34, 139, 34),  "shape": "triangle"},
    "Sports Shoes":  {"color": (255, 140, 0),  "shape": "diamond"},
    "Dresses":       {"color": (148, 0, 211),  "shape": "hexagon"},
    "Handbags":      {"color": (184, 134, 11), "shape": "rectangle_rounded"},
}


def _draw_shape(draw, shape, box, color):
    x0, y0, x1, y1 = box
    if shape == "rectangle":
        draw.rectangle(box, fill=color)
    elif shape == "rectangle_rounded":
        draw.rounded_rectangle(box, radius=20, fill=color)
    elif shape == "ellipse":
        draw.ellipse(box, fill=color)
    elif shape == "triangle":
        draw.polygon([(x0, y1), ((x0 + x1) // 2, y0), (x1, y1)], fill=color)
    elif shape == "diamond":
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=color)
    elif shape == "hexagon":
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        w, h = (x1 - x0) // 2, (y1 - y0) // 2
        pts = [
            (cx - w, cy), (cx - w // 2, cy - h), (cx + w // 2, cy - h),
            (cx + w, cy), (cx + w // 2, cy + h), (cx - w // 2, cy + h),
        ]
        draw.polygon(pts, fill=color)


def make_image(category, idx, size=224, seed=0):
    rng = np.random.RandomState(seed + idx)
    cfg = CATEGORIES[category]
    base_color = np.array(cfg["color"])
    # jitter color and shape position/size slightly per-image so images in
    # the same category are similar-but-not-identical (more realistic)
    jitter = rng.randint(-25, 25, size=3)
    color = tuple(np.clip(base_color + jitter, 0, 255).tolist())
    bg = tuple(rng.randint(230, 255, size=3).tolist())

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    margin = rng.randint(20, 50)
    box = (margin, margin, size - margin, size - margin)
    _draw_shape(draw, cfg["shape"], box, color)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_category", type=int, default=30)
    parser.add_argument("--out_dir", type=str, default=SAMPLE_DATA_DIR)
    args = parser.parse_args()

    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    for category in CATEGORIES:
        for i in range(args.per_category):
            img = make_image(category, i, seed=SEED)
            fname = f"{category.replace(' ', '_')}_{i:04d}.jpg"
            img.save(os.path.join(images_dir, fname), quality=90)
            split = "train" if i < int(args.per_category * 0.7) else (
                "val" if i < int(args.per_category * 0.85) else "test"
            )
            rows.append({"image_path": os.path.join("images", fname),
                         "category": category, "split": split})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "labels.csv"), index=False)
    print(f"Generated {len(df)} synthetic images across {len(CATEGORIES)} categories at {args.out_dir}")


if __name__ == "__main__":
    main()
