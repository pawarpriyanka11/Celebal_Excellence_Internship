from __future__ import annotations

import numpy as np


def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def peak_signal_to_noise_ratio(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def improvement_percentage(noisy_mse: float, denoised_mse: float) -> float:
    if noisy_mse <= 0:
        return 0.0
    return float(((noisy_mse - denoised_mse) / noisy_mse) * 100.0)
