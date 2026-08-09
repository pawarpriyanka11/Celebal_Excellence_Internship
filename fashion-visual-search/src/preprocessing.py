"""Image preprocessing: resize to 224x224 and normalize with ImageNet stats."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def load_image(path: str) -> Image.Image:
    """Load an image file as RGB."""
    return Image.open(path).convert("RGB")


def resize_image(img: Image.Image, size=IMG_SIZE) -> Image.Image:
    return img.resize(size, Image.BILINEAR)


def normalize(arr: np.ndarray) -> np.ndarray:
    """arr: float array in [0, 1], shape (H, W, 3). Returns ImageNet-normalized array."""
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (arr - mean) / std


def preprocess_image(path: str, size=IMG_SIZE) -> np.ndarray:
    """Full pipeline: load -> resize -> scale to [0,1] -> ImageNet-normalize.

    Returns a (H, W, 3) float32 array ready to be batched and fed to the model.
    """
    img = resize_image(load_image(path), size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return normalize(arr)


def preprocess_batch(paths, base_dir: str = "", size=IMG_SIZE) -> np.ndarray:
    """Preprocess a list of image paths (relative to base_dir) into a batch array."""
    batch = np.zeros((len(paths), size[0], size[1], 3), dtype=np.float32)
    for i, p in enumerate(paths):
        full_path = os.path.join(base_dir, p) if base_dir else p
        batch[i] = preprocess_image(full_path, size)
    return batch
