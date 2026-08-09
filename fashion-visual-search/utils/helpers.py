"""Shared helper utilities used across the pipeline."""
import os
import random
import time
import numpy as np
import pandas as pd


def set_seed(seed: int = 42):
    """Make runs reproducible across numpy / python / tensorflow (if present)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


class Timer:
    """Simple context-manager timer.

    with Timer() as t:
        do_something()
    print(t.elapsed)
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start


def load_labels(csv_path: str) -> pd.DataFrame:
    """Load a labels.csv with columns [image_path, category, split?]."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Labels file not found at {csv_path}. "
            "Run data/prepare_subset.py (or data/make_sample_data.py) first."
        )
    return pd.read_csv(csv_path)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms
