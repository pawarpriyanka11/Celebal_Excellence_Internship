"""
Transfer learning stage: freeze most of a pretrained ResNet50, add a small
classification head, and fine-tune the last few backbone layers + head on
the category-labeled subset. The classifier is only a means to an end here —
what we actually want is the resulting *embedding* layer (the penultimate
dense layer), which becomes the "finetuned" backbone used by
feature_extraction.py.

Usage:
    python src/transfer_learning.py --data_dir data/subset --epochs 10 --out models/finetuned_backbone.h5
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    IMG_SIZE, EMBEDDING_DIM, TL_EPOCHS, TL_BATCH_SIZE, TL_LR, TL_UNFREEZE_LAST_N, SEED,
)
from utils.helpers import load_labels, set_seed  # noqa: E402
from src.preprocessing import preprocess_batch  # noqa: E402


def build_model(num_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", pooling="avg", input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = True
    # freeze all but the last N layers
    for layer in base.layers[:-TL_UNFREEZE_LAST_N]:
        layer.trainable = False

    embedding = layers.Dense(EMBEDDING_DIM, activation="relu", name="embedding")(base.output)
    embedding = layers.BatchNormalization()(embedding)
    logits = layers.Dense(num_classes, activation="softmax", name="classifier")(embedding)

    full_model = models.Model(base.input, logits, name="finetuned_classifier")
    embedding_model = models.Model(base.input, embedding, name="finetuned_embedding")
    return full_model, embedding_model


class NpyDataGenerator:
    """Simple batch generator that preprocesses images on the fly (keeps RAM low)."""

    def __init__(self, df, data_dir, label_to_idx, batch_size, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label_to_idx = label_to_idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(df)

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def generator(self):
        import tensorflow as tf
        idxs = np.arange(self.n)
        while True:
            if self.shuffle:
                np.random.shuffle(idxs)
            for i in range(0, self.n, self.batch_size):
                batch_idx = idxs[i:i + self.batch_size]
                rows = self.df.iloc[batch_idx]
                X = preprocess_batch(rows["image_path"].tolist(), base_dir=self.data_dir)
                y = np.array([self.label_to_idx[c] for c in rows["category"]])
                y = tf.keras.utils.to_categorical(y, num_classes=len(self.label_to_idx))
                yield X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=TL_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=TL_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TL_LR)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    set_seed(SEED)
    import tensorflow as tf

    df = load_labels(os.path.join(args.data_dir, "labels.csv"))
    categories = sorted(df["category"].unique())
    label_to_idx = {c: i for i, c in enumerate(categories)}

    train_df = df[df["split"] == "train"] if "split" in df else df.sample(frac=0.8, random_state=SEED)
    val_df = df[df["split"] == "val"] if "split" in df else df.drop(train_df.index)

    full_model, embedding_model = build_model(num_classes=len(categories))
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    full_model.summary()

    train_gen = NpyDataGenerator(train_df, args.data_dir, label_to_idx, args.batch_size, shuffle=True)
    val_gen = NpyDataGenerator(val_df, args.data_dir, label_to_idx, args.batch_size, shuffle=False)

    full_model.fit(
        train_gen.generator(),
        steps_per_epoch=max(len(train_gen), 1),
        validation_data=val_gen.generator(),
        validation_steps=max(len(val_gen), 1),
        epochs=args.epochs,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    embedding_model.save(args.out)

    labels_path = args.out.replace(".h5", "_labels.txt").replace(".keras", "_labels.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(categories))

    print(f"Saved fine-tuned embedding model to {args.out}")
    print(f"Category order saved to {labels_path}")


if __name__ == "__main__":
    main()
