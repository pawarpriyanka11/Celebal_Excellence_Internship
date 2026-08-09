"""
Feature extraction: turns a folder of images into L2-normalized embedding
vectors using one of three backbones:

  --model resnet50    : pretrained ResNet50 (ImageNet weights), GAP output, no head    (baseline)
  --model efficientnet: pretrained EfficientNetB0, GAP output, no head                 (baseline alt.)
  --model finetuned   : ResNet50 backbone + fine-tuned last layers (see transfer_learning.py)
  --model siamese      : Siamese-trained embedding backbone (see train_siamese.py)

Saves an .npz with arrays: embeddings (N, D), image_paths (N,), categories (N,)

Usage:
    python src/feature_extraction.py --data_dir data/subset --model resnet50 --out models/embeddings/baseline.npz
"""
import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE  # noqa: E402
from utils.helpers import load_labels, l2_normalize, Timer  # noqa: E402
from src.preprocessing import preprocess_batch  # noqa: E402
from src.siamese_model import load_embedding_branch  # noqa: E402


def build_backbone(model_name: str, weights_path: str = None):
    """Returns a keras.Model mapping a preprocessed image batch -> embedding batch."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    if model_name == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", pooling="avg",
            input_shape=(*IMG_SIZE, 3),
        )
        base.trainable = False
        return models.Model(base.input, base.output, name="resnet50_baseline")

    if model_name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", pooling="avg",
            input_shape=(*IMG_SIZE, 3),
        )
        base.trainable = False
        return models.Model(base.input, base.output, name="efficientnet_baseline")

    if model_name in ("finetuned", "siamese"):
        if weights_path is None or not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"--weights is required and must exist for model='{model_name}'. "
                f"Got: {weights_path}. Train it first with transfer_learning.py / train_siamese.py"
            )

        if model_name == "siamese":
            # The Siamese embedding branch ends with a Lambda layer, so it is
            # persisted as weights-only (see train_siamese.py). Rebuild the
            # architecture and load the trained weights to avoid the Keras/TF
            # 2.16 HDF5 (Lambda) serialization incompatibility.
            return load_embedding_branch(weights_path)

        # The finetuned model is a full embedding model (see transfer_learning.py) —
        # just load and reuse.
        return tf.keras.models.load_model(weights_path, compile=False)

    raise ValueError(f"Unknown model '{model_name}'")


def extract_embeddings(data_dir: str, model_name: str, weights_path: str = None,
                        batch_size: int = 32, labels_csv: str = None):
    labels_csv = labels_csv or os.path.join(data_dir, "labels.csv")
    df = load_labels(labels_csv)

    backbone = build_backbone(model_name, weights_path)

    all_embeddings = []
    paths = df["image_path"].tolist()
    with Timer() as t:
        for i in tqdm(range(0, len(paths), batch_size), desc=f"Extracting ({model_name})"):
            batch_paths = paths[i:i + batch_size]
            batch = preprocess_batch(batch_paths, base_dir=data_dir)
            emb = backbone.predict(batch, verbose=0)
            all_embeddings.append(emb)
    embeddings = np.concatenate(all_embeddings, axis=0)
    embeddings = l2_normalize(embeddings)

    print(f"Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]} "
          f"in {t.elapsed:.2f}s ({t.elapsed / max(len(paths),1)*1000:.1f} ms/image)")

    return embeddings, df["image_path"].values, df["category"].values, (df["split"].values if "split" in df else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="resnet50",
                         choices=["resnet50", "efficientnet", "finetuned", "siamese"])
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    embeddings, paths, categories, splits = extract_embeddings(
        args.data_dir, args.model, args.weights, args.batch_size
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, embeddings=embeddings, image_paths=paths, categories=categories,
              splits=splits if splits is not None else np.array([]), model_name=args.model,
              data_dir=args.data_dir)
    print(f"Saved embeddings to {args.out}")


if __name__ == "__main__":
    main()
