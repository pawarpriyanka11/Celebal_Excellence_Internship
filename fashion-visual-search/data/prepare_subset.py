"""
Builds the curated training subset described in the requirements:
  - Select 5-8 categories (articleType) from styles.csv
  - Sample ~200-300 images per category (~1500-2500 total)
  - Copy the corresponding images into data/subset/images/
  - Write data/subset/labels.csv with a stratified train/val/test split

Usage:
    python data/prepare_subset.py \
        --categories "Shirts,Tshirts,Casual Shoes,Sports Shoes,Dresses,Handbags,Watches,Sunglasses" \
        --per_category 250
"""
import argparse
import os
import shutil
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    RAW_DATA_DIR, SUBSET_DIR, SUBSET_IMAGES_DIR, SUBSET_LABELS_CSV,
    DEFAULT_CATEGORIES, IMAGES_PER_CATEGORY, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SEED,
)


def build_subset(categories, per_category, styles_csv, images_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    styles = pd.read_csv(styles_csv, on_bad_lines="skip")
    styles = styles[styles["articleType"].isin(categories)]

    rows = []
    for cat in categories:
        cat_df = styles[styles["articleType"] == cat]
        n = min(per_category, len(cat_df))
        if n < per_category:
            print(f"[warn] only {n} images available for '{cat}' (< {per_category})")
        sampled = cat_df.sample(n=n, random_state=SEED)
        for _, row in sampled.iterrows():
            src = os.path.join(images_dir, f"{row['id']}.jpg")
            if not os.path.exists(src):
                continue
            dst_name = f"{row['id']}.jpg"
            dst = os.path.join(out_dir, "images", dst_name)
            shutil.copyfile(src, dst)
            rows.append({"image_path": os.path.join("images", dst_name), "category": cat})

    df = pd.DataFrame(rows)

    # stratified train/val/test split
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_SPLIT + TEST_SPLIT), stratify=df["category"], random_state=SEED
    )
    rel_test = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["category"], random_state=SEED
    )
    train_df["split"], val_df["split"], test_df["split"] = "train", "val", "test"

    final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    final_df.to_csv(os.path.join(out_dir, "labels.csv"), index=False)
    return final_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=str, default=",".join(DEFAULT_CATEGORIES),
                         help="Comma-separated list of articleType categories")
    parser.add_argument("--per_category", type=int, default=IMAGES_PER_CATEGORY)
    parser.add_argument("--styles_csv", type=str, default=os.path.join(RAW_DATA_DIR, "styles.csv"))
    parser.add_argument("--images_dir", type=str, default=os.path.join(RAW_DATA_DIR, "images"))
    parser.add_argument("--out_dir", type=str, default=SUBSET_DIR)
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")]
    if not (5 <= len(categories) <= 8):
        print(f"[warn] requirements suggest 5-8 categories, got {len(categories)}")

    if not os.path.exists(args.styles_csv):
        print(f"styles.csv not found at {args.styles_csv}.")
        print("Run 'python data/download_data.py' first, or use sample_data/ for a smoke test.")
        sys.exit(1)

    df = build_subset(categories, args.per_category, args.styles_csv, args.images_dir, args.out_dir)
    print(f"Subset built: {len(df)} images across {df['category'].nunique()} categories")
    print(df["category"].value_counts())
    print(f"Saved to {args.out_dir}/labels.csv")


if __name__ == "__main__":
    main()
