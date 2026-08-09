"""
Downloads the Fashion Product Images Dataset from Kaggle.

Requires Kaggle API credentials at ~/.kaggle/kaggle.json
(create one at https://www.kaggle.com/settings -> API -> Create New Token).

Usage:
    python data/download_data.py
"""
import os
import sys
import zipfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR  # noqa: E402

DATASET_SLUG = "paramaggarwal/fashion-product-images-dataset"


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("The 'kaggle' package is not installed. Run: pip install kaggle")
        sys.exit(1)
    except OSError as e:
        print(
            "Kaggle credentials not found. Place kaggle.json in ~/.kaggle/ "
            "or set KAGGLE_USERNAME / KAGGLE_KEY env vars.\n"
            f"Original error: {e}"
        )
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET_SLUG} into {RAW_DATA_DIR} ...")
    api.dataset_download_files(DATASET_SLUG, path=RAW_DATA_DIR, quiet=False)

    zip_path = os.path.join(RAW_DATA_DIR, "fashion-product-images-dataset.zip")
    if os.path.exists(zip_path):
        print("Unzipping...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DATA_DIR)
        os.remove(zip_path)

    print("Done. Expected structure:")
    print(f"  {RAW_DATA_DIR}/images/*.jpg")
    print(f"  {RAW_DATA_DIR}/styles.csv")
    print("Next: python data/prepare_subset.py")


if __name__ == "__main__":
    main()
