from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


def validate_uploaded_file(file: Any, file_size: int) -> None:
    if file is None:
        raise ValueError("No image uploaded.")
    if file.filename is None or file.filename.strip() == "":
        raise ValueError("No image uploaded.")
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise ValueError("Unsupported file type. Please upload a PNG or JPG image.")
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("File too large. Maximum size is 5 MB.")


def open_and_validate_image(file: Any) -> Image.Image:
    file.file.seek(0)
    try:
        image = Image.open(file.file)
        image.verify()
        file.file.seek(0)
        image = Image.open(file.file)
        image = image.convert("RGB")
        if image.width == 0 or image.height == 0:
            raise ValueError("The uploaded image is invalid.")
        return image
    except Exception as exc:
        raise ValueError("Could not read the uploaded image. Please upload a valid PNG or JPG file.") from exc


def preprocess_image(image: Image.Image) -> np.ndarray:
    grayscale = image.convert("L")
    source = np.asarray(grayscale, dtype=np.float32) / 255.0

    border = np.concatenate((source[0], source[-1], source[:, 0], source[:, -1]))
    if float(np.median(border)) > 0.5:
        source = 1.0 - source

    source = np.clip(source, 0.0, 1.0)
    rows, cols = np.where(source > 0.12)
    if rows.size == 0 or cols.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    cropped = source[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    crop_image = Image.fromarray(np.uint8(cropped * 255.0), mode="L")
    crop_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - crop_image.width) // 2
    top = (28 - crop_image.height) // 2
    canvas.paste(crop_image, (left, top))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def add_gaussian_noise(array: np.ndarray, noise_factor: float) -> np.ndarray:
    noise = np.random.normal(0.0, noise_factor, array.shape).astype(np.float32)
    noisy = np.clip(array + noise, 0.0, 1.0)
    return noisy


def image_to_base64(image_array: np.ndarray) -> str:
    image = Image.fromarray(np.uint8(image_array * 255.0))
    output_path = Path("/tmp") / "temp.png"
    image.save(output_path, format="PNG")
    return output_path.read_bytes()
