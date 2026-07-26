from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D


MODEL_PATH = Path(__file__).resolve().parent / "models" / "mnist_denoising_autoencoder.keras"


def build_autoencoder() -> Model:
    input_layer = Input(shape=(28, 28, 1), name="input_image")
    encoded = Conv2D(32, (3, 3), activation="relu", padding="same", name="encode_conv_32")(input_layer)
    encoded = MaxPooling2D((2, 2), padding="same", name="encode_pool_1")(encoded)
    encoded = Conv2D(16, (3, 3), activation="relu", padding="same", name="encode_conv_16")(encoded)
    encoded = MaxPooling2D((2, 2), padding="same", name="latent_space")(encoded)
    decoded = Conv2D(16, (3, 3), activation="relu", padding="same", name="decode_conv_16")(encoded)
    decoded = UpSampling2D((2, 2), name="decode_upsample_1")(decoded)
    decoded = Conv2D(32, (3, 3), activation="relu", padding="same", name="decode_conv_32")(decoded)
    decoded = UpSampling2D((2, 2), name="decode_upsample_2")(decoded)
    output = Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="output_image")(decoded)
    model = Model(inputs=input_layer, outputs=output, name="mnist_denoising_autoencoder")
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def train_and_save_model() -> None:
    (x_train, _), _ = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    rng = np.random.default_rng(42)
    noise_levels = rng.uniform(0.2, 0.45, size=(x_train.shape[0], 1, 1, 1)).astype("float32")
    noisy_train = np.clip(x_train + rng.normal(0.0, noise_levels, x_train.shape), 0.0, 1.0).astype("float32")
    model = build_autoencoder()
    model.fit(
        noisy_train,
        x_train,
        epochs=10,
        batch_size=128,
        shuffle=True,
        validation_split=0.1,
        verbose=1,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()
