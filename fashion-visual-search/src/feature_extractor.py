"""
Step 2 & 3: Feature Extraction + Baseline Similarity
------------------------------------------------------
Loads a pretrained CNN (EfficientNetB0 by default, ImageNet weights,
classification head removed) as a fixed feature extractor, embeds every
image in the subset, and builds both:
  - a FAISS index (fast retrieval)
  - a plain sklearn cosine-similarity function (explicit baseline)

This module is reused by the Siamese pipeline (same preprocessing).
"""
import hashlib
import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

import config

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Backbone-specific imports
if config.BACKBONE == "efficientnet":
    from tensorflow.keras.applications.efficientnet import (
        EfficientNetB0, preprocess_input,
    )
else:
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2, preprocess_input,
    )


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_backbone(trainable=False):
    """EfficientNetB0 (default) or MobileNetV2 with classification head
    removed, global-average-pooled to a single embedding vector."""
    kwargs = dict(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    if config.BACKBONE == "efficientnet":
        base = EfficientNetB0(**kwargs)
    else:
        base = MobileNetV2(**kwargs)
    base.trainable = trainable
    return base


# ----------------------------------------------------------------------
# Image loading / preprocessing
# ----------------------------------------------------------------------
def load_and_preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)  # ImageNet normalization
    return arr


def embed_single_image(model, pil_image) -> np.ndarray:
    """Embed a single PIL image and return an L2-normalized vector.
    Shared by feature_extractor and streamlit_app to avoid duplication."""
    img = pil_image.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    emb = model.predict(arr, verbose=0)[0]
    norm = np.linalg.norm(emb)
    return (emb / (norm if norm > 0 else 1e-8)).astype("float32")


def batch_generator(image_paths, batch_size=32):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch = np.stack([load_and_preprocess(p) for p in batch_paths])
        yield batch


# ----------------------------------------------------------------------
# Embedding extraction
# ----------------------------------------------------------------------
def extract_embeddings(model, image_paths, batch_size=32, verbose=True):
    all_embeddings = []
    n_batches = (len(image_paths) + batch_size - 1) // batch_size
    for i, batch in enumerate(batch_generator(image_paths, batch_size)):
        emb = model.predict(batch, verbose=0)
        all_embeddings.append(emb)
        if verbose and (i + 1) % 5 == 0:
            print(f"  embedded batch {i + 1}/{n_batches}")
    embeddings = np.vstack(all_embeddings)
    # L2-normalize so cosine similarity == dot product (needed for FAISS IP index)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = embeddings / norms
    return embeddings.astype("float32")


# ----------------------------------------------------------------------
# FAISS index
# ----------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray):
    if not FAISS_AVAILABLE:
        raise ImportError("faiss-cpu is not installed. `pip install faiss-cpu`")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
    index.add(embeddings)
    return index


def search_faiss(index, query_embedding: np.ndarray, top_k=10):
    query_embedding = query_embedding.reshape(1, -1).astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    return indices[0], scores[0]


# ----------------------------------------------------------------------
# Plain sklearn cosine-similarity baseline (no FAISS dependency)
# ----------------------------------------------------------------------
def search_cosine_sklearn(embeddings: np.ndarray, query_embedding: np.ndarray, top_k=10):
    query_embedding = query_embedding.reshape(1, -1)
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(-sims)[:top_k]
    return top_indices, sims[top_indices]


# ----------------------------------------------------------------------
# Subset hash — skip re-extraction if subset.csv hasn't changed
# ----------------------------------------------------------------------
def _subset_hash(csv_path: str) -> str:
    with open(csv_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _hash_path(emb_path: str) -> str:
    return emb_path.replace(".npy", ".md5")


def embeddings_are_fresh(csv_path: str, emb_path: str) -> bool:
    hp = _hash_path(emb_path)
    if not (os.path.exists(emb_path) and os.path.exists(hp)):
        return False
    with open(hp) as f:
        return f.read().strip() == _subset_hash(csv_path)


def save_hash(csv_path: str, emb_path: str):
    with open(_hash_path(emb_path), "w") as f:
        f.write(_subset_hash(csv_path))


# ----------------------------------------------------------------------
# Pipeline entry point: build baseline embeddings + index for the subset
# ----------------------------------------------------------------------
def main():
    if not os.path.exists(config.SUBSET_CSV):
        raise FileNotFoundError(
            f"{config.SUBSET_CSV} not found. Run `python src/data_prep.py` first."
        )

    if embeddings_are_fresh(config.SUBSET_CSV, config.BASELINE_EMB_PATH):
        print("Baseline embeddings are up-to-date (subset.csv unchanged). Skipping.")
        return

    df = pd.read_csv(config.SUBSET_CSV)
    print(f"Extracting baseline ({config.BACKBONE}, ImageNet weights) embeddings "
          f"for {len(df)} images ...")

    model = build_backbone(trainable=False)
    embeddings = extract_embeddings(model, df["image_path"].tolist())

    np.save(config.BASELINE_EMB_PATH, embeddings)
    df.to_csv(config.BASELINE_META_PATH, index=False)
    save_hash(config.SUBSET_CSV, config.BASELINE_EMB_PATH)
    print(f"Saved embeddings -> {config.BASELINE_EMB_PATH}  shape={embeddings.shape}")

    if FAISS_AVAILABLE:
        index = build_faiss_index(embeddings)
        faiss.write_index(index, config.BASELINE_FAISS_PATH)
        print(f"Saved FAISS index -> {config.BASELINE_FAISS_PATH}")
    else:
        print("faiss-cpu not installed - skipping FAISS index "
              "(sklearn cosine similarity will still work).")


if __name__ == "__main__":
    main()
