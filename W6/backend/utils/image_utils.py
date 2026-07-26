from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def array_to_png_base64(image_array: np.ndarray) -> str:
    image = Image.fromarray(np.uint8(image_array * 255.0), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    binary = buffer.getvalue()
    return base64.b64encode(binary).decode("utf-8")


def build_comparison_image(original: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> str:
    width = 28 * 3 + 5
    canvas = Image.new("L", (width, 28), color=255)
    for index, array in enumerate((original, noisy, denoised)):
        image = Image.fromarray(np.uint8(array * 255.0), mode="L")
        canvas.paste(image, (index * 28 + index * 2, 0))
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
