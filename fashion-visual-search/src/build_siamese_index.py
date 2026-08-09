"""
Builds embeddings + FAISS index for the whole subset using the
fine-tuned Siamese embedding model (mirrors feature_extractor.py's
baseline pipeline so the two are directly comparable in evaluate.py).

Run (after train_siamese.py has produced the saved model):
    python src/build_siamese_index.py
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from siamese_model import L2Normalize  # noqa: F401 - registers custom layer
from feature_extractor import batch_generator, build_faiss_index

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def extract_siamese_embeddings(model, image_paths, batch_size=32):
    all_embeddings = []
    for i, batch in enumerate(batch_generator(image_paths, batch_size)):
        emb = model.predict(batch, verbose=0)
        all_embeddings.append(emb)
        if (i + 1) % 5 == 0:
            print(f"  embedded batch {i + 1}")
    embeddings = np.vstack(all_embeddings)
    # embedding network already L2-normalizes internally, but re-normalize
    # defensively in case of float drift
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return (embeddings / norms).astype("float32")


def main():
    if not os.path.exists(config.SIAMESE_MODEL_PATH):
        raise FileNotFoundError(
            f"{config.SIAMESE_MODEL_PATH} not found. Run "
            "`python src/train_siamese.py` first."
        )
    if not os.path.exists(config.SUBSET_CSV):
        raise FileNotFoundError(
            f"{config.SUBSET_CSV} not found. Run `python src/data_prep.py` first."
        )

    df = pd.read_csv(config.SUBSET_CSV)
    model = tf.keras.models.load_model(config.SIAMESE_MODEL_PATH, compile=False, safe_mode=False)

    print(f"Extracting Siamese fine-tuned embeddings for {len(df)} images ...")
    embeddings = extract_siamese_embeddings(model, df["image_path"].tolist())

    np.save(config.SIAMESE_EMB_PATH, embeddings)
    df.to_csv(config.SIAMESE_META_PATH, index=False)
    print(f"Saved embeddings -> {config.SIAMESE_EMB_PATH}  shape={embeddings.shape}")

    if FAISS_AVAILABLE:
        index = build_faiss_index(embeddings)
        faiss.write_index(index, config.SIAMESE_FAISS_PATH)
        print(f"Saved FAISS index -> {config.SIAMESE_FAISS_PATH}")


if __name__ == "__main__":
    main()
