from __future__ import annotations

import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"

if __name__ == "__main__" and VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON:
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from utils.image_utils import array_to_png_base64, build_comparison_image
from utils.metrics import improvement_percentage, mean_squared_error, peak_signal_to_noise_ratio
from utils.preprocessing import add_gaussian_noise, open_and_validate_image, preprocess_image, validate_uploaded_file

MODEL_PATH = BASE_DIR / "models" / "mnist_denoising_autoencoder.keras"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mnist_denoise_api")

model = None


def startup_event() -> None:
    global model
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        import keras
        model = keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    except Exception as exc:
        logger.exception("Model loading failed")
        raise RuntimeError("Model loading failed") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup_event()
    yield


app = FastAPI(title="MNIST Denoise AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/api/model-info")
def model_info() -> dict:
    return {
        "name": "mnist_denoising_autoencoder",
        "architecture": "Convolutional Autoencoder",
        "input": "28 × 28 grayscale image",
        "latent_dimension": "7 x 7 x 16",
        "activation": "ReLU + Sigmoid",
        "optimizer": "Adam",
        "loss_function": "Binary Cross-Entropy",
        "dataset": "MNIST",
        "task": "Image Denoising",
    }


@app.post("/api/denoise")
def denoise(image: UploadFile = File(...), noise_factor: float = Form(...)) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not ready yet.")

    try:
        if noise_factor < 0 or noise_factor > 0.6:
            raise ValueError("Noise factor must be between 0 and 0.6.")

        file_size = image.file.seek(0, os.SEEK_END)
        image.file.seek(0)
        validate_uploaded_file(image, file_size)
        pillow_image = open_and_validate_image(image)
        original_array = preprocess_image(pillow_image)
        noisy_array = add_gaussian_noise(original_array, noise_factor)

        batch = noisy_array[np.newaxis, ..., np.newaxis].astype(np.float32)
        prediction = model.predict(batch, verbose=0)
        denoised = np.clip(np.squeeze(prediction, axis=(0, 3)), 0.0, 1.0)

        noisy_mse = mean_squared_error(original_array, noisy_array)
        denoised_mse = mean_squared_error(original_array, denoised)
        psnr = peak_signal_to_noise_ratio(denoised_mse)
        improvement = improvement_percentage(noisy_mse, denoised_mse)

        return {
            "success": True,
            "original_image": array_to_png_base64(original_array),
            "noisy_image": array_to_png_base64(noisy_array),
            "denoised_image": array_to_png_base64(denoised),
            "comparison_image": build_comparison_image(original_array, noisy_array, denoised),
            "metrics": {
                "noisy_mse": round(noisy_mse, 6),
                "denoised_mse": round(denoised_mse, 6),
                "psnr": round(psnr, 4),
                "improvement_percentage": round(improvement, 4),
            },
        }
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
