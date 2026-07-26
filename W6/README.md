# MNIST Denoise AI

## Overview

MNIST Denoise AI is a deep learning image restoration demo that uses a trained autoencoder to remove Gaussian noise from MNIST-style handwritten digit images.

## Features

- Upload a handwritten digit image
- Preview original and noisy inputs
- Run denoising through a TensorFlow autoencoder on the backend
- View quality metrics such as MSE, PSNR, and improvement percentage
- Download the denoised image and comparison output

## Architecture

- Frontend: React + Vite
- Backend: FastAPI + TensorFlow + Pillow + NumPy
- Model: `backend/models/mnist_denoising_autoencoder.keras`

## ML model explanation

The model is a fully connected autoencoder trained to reconstruct clean 28 × 28 grayscale digits from noisy inputs.

## Project structure

- `backend/` contains the FastAPI API and model assets
- `frontend/` contains the React dashboard UI

## Installation

### Backend

```bash
cd backend
py -m pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Run

1. Start the backend on port 8000.
2. Start the frontend dev server.
3. Visit the frontend URL shown by Vite.

## API

- `GET /api/health`
- `GET /api/model-info`
- `POST /api/denoise`

## Future improvements

- Add more model variants and training visualization
- Support larger custom datasets
- Add comparison slider enhancements
