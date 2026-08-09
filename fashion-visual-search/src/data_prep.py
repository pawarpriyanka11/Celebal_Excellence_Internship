"""
Step 1: Dataset Preparation
----------------------------
Reads the full Kaggle "Fashion Product Images Dataset" styles.csv,
selects SELECTED_CATEGORIES (5-8 categories), samples ~200-300 images
per category, copies those images into data/subset/images, and writes
data/subset/subset.csv with a train/query split column for evaluation.

Run:
    python src/data_prep.py
"""
import os
import shutil
import pandas as pd
from PIL import Image

import config


def load_styles_csv() -> pd.DataFrame:
    if not os.path.exists(config.STYLES_CSV):
        raise FileNotFoundError(
            f"Could not find {config.STYLES_CSV}.\n"
            "Download the dataset from Kaggle first:\n"
            "  kaggle datasets download -d paramaggarwal/fashion-product-images-dataset\n"
            "then unzip so that data/raw/styles.csv and data/raw/images/ exist."
        )
    # the Kaggle csv occasionally has a few malformed rows -> skip them
    df = pd.read_csv(config.STYLES_CSV, on_bad_lines="skip")
    return df


def build_subset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["articleType"].isin(config.SELECTED_CATEGORIES)].copy()

    # keep only rows whose image file actually exists on disk
    df["image_filename"] = df["id"].astype(str) + ".jpg"
    df["src_path"] = df["image_filename"].apply(
        lambda f: os.path.join(config.RAW_IMAGES_DIR, f)
    )
    df = df[df["src_path"].apply(os.path.exists)]

    sampled_frames = []
    for category in config.SELECTED_CATEGORIES:
        cat_df = df[df["articleType"] == category]
        n = min(config.SAMPLES_PER_CATEGORY, len(cat_df))
        if n == 0:
            print(f"[WARN] No images found for category '{category}', skipping.")
            continue
        sampled = cat_df.sample(n=n, random_state=config.RANDOM_SEED)
        sampled_frames.append(sampled)
        print(f"  {category:15s} -> sampled {n} images (available: {len(cat_df)})")

    subset_df = pd.concat(sampled_frames, ignore_index=True)
    return subset_df


def copy_and_validate_images(subset_df: pd.DataFrame) -> pd.DataFrame:
    """Copies sampled images into data/subset/images, drops any that
    fail to open (corrupted files exist in this dataset)."""
    valid_rows = []
    for _, row in subset_df.iterrows():
        dst_path = os.path.join(config.SUBSET_IMAGES_DIR, row["image_filename"])
        try:
            with Image.open(row["src_path"]) as im:
                im.convert("RGB")  # validates the file decodes correctly
            shutil.copyfile(row["src_path"], dst_path)
            row = row.copy()
            row["image_path"] = dst_path
            valid_rows.append(row)
        except Exception as e:
            print(f"  [SKIP] corrupted image {row['src_path']}: {e}")

    return pd.DataFrame(valid_rows)


def add_query_split(df: pd.DataFrame) -> pd.DataFrame:
    """Marks a held-out 'query' fraction per category, used later to
    evaluate Precision@K / Recall@K honestly (queries are never counted
    as their own match at retrieval time)."""
    df = df.sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
    df["split"] = "index"
    for category in config.SELECTED_CATEGORIES:
        idx = df[df["articleType"] == category].index
        n_query = max(1, int(len(idx) * config.QUERY_FRACTION))
        query_idx = idx[:n_query]
        df.loc[query_idx, "split"] = "query"
    return df


def main():
    print("Loading styles.csv ...")
    df = load_styles_csv()
    print(f"  total rows in full dataset: {len(df)}")

    print(f"\nBuilding subset from categories: {config.SELECTED_CATEGORIES}")
    subset_df = build_subset(df)

    print(f"\nCopying + validating {len(subset_df)} images -> {config.SUBSET_IMAGES_DIR}")
    subset_df = copy_and_validate_images(subset_df)

    subset_df = add_query_split(subset_df)

    keep_cols = [
        "id", "image_filename", "image_path", "gender", "masterCategory",
        "subCategory", "articleType", "baseColour", "season", "year",
        "usage", "productDisplayName", "split",
    ]
    keep_cols = [c for c in keep_cols if c in subset_df.columns]
    subset_df = subset_df[keep_cols]

    subset_df.to_csv(config.SUBSET_CSV, index=False)

    print(f"\nDone. Final subset size: {len(subset_df)} images")
    print(subset_df["articleType"].value_counts())
    print(f"\nIndex/query split:\n{subset_df['split'].value_counts()}")
    print(f"\nSaved subset metadata -> {config.SUBSET_CSV}")


if __name__ == "__main__":
    main()
