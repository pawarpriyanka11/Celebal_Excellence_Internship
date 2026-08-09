"""
Step 5 & 6: Transfer Learning + Siamese Network Training
------------------------------------------------------------
Stage 1 (transfer learning warm-up, optional but recommended):
    Freeze most of MobileNetV2, train a lightweight classification head
    on the subset's articleType labels for a couple epochs. This nudges
    the unfrozen last layers towards this dataset's visual style before
    triplet training starts, which stabilizes triplet loss convergence.

Stage 2 (core enhancement):
    Reuse the same partially-unfrozen backbone as the shared tower of a
    Siamese network, trained with triplet loss on (anchor, positive,
    negative) triplets built from category labels.

Run:
    python src/train_siamese.py
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

import config
from feature_extractor import build_backbone, load_and_preprocess
from siamese_model import (
    build_embedding_network,
    build_siamese_training_model,
    triplet_loss,
)
from triplet_data_generator import TripletGenerator


def stage1_transfer_learning_warmup(df: pd.DataFrame):
    """Quick classification warm-up on articleType, using the same
    partially-unfrozen backbone that will feed into the Siamese tower.
    Returns the fitted backbone (weights are what matter, head is discarded)."""
    print("\n=== Stage 1: transfer-learning warm-up (classification head) ===")

    train_df = df[df["split"] == "index"].reset_index(drop=True)
    categories = sorted(train_df["articleType"].unique())
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    labels = train_df["articleType"].map(cat_to_idx).to_numpy()

    print(f"  {len(train_df)} training images across {len(categories)} categories")

    backbone = build_backbone(trainable=True)
    for layer in backbone.layers[:-config.UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False

    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS))
    x = backbone(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(categories), activation="softmax")(x)
    clf_model = Model(inputs, outputs)

    clf_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_FINE_TUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    def gen():
        rng = np.random.default_rng(config.RANDOM_SEED)
        n = len(train_df)
        while True:
            batch_idx = rng.choice(n, size=config.BATCH_SIZE, replace=False)
            batch_imgs = np.stack([
                load_and_preprocess(train_df.loc[i, "image_path"]) for i in batch_idx
            ])
            batch_labels = labels[batch_idx]
            yield batch_imgs, batch_labels

    clf_model.fit(
        gen(),
        steps_per_epoch=config.STEPS_PER_EPOCH,
        epochs=config.EPOCHS_FINE_TUNE,
        verbose=1,
    )

    return backbone  # fine-tuned weights carry over into the Siamese tower


def stage2_siamese_triplet_training(df: pd.DataFrame, warmed_backbone=None):
    print("\n=== Stage 2: Siamese network training (triplet loss) ===")

    embedding_model = build_embedding_network()

    if warmed_backbone is not None:
        # transplant the stage-1 fine-tuned backbone weights into the
        # embedding network's backbone sub-layer before triplet training
        embedding_model.layers[1].set_weights(warmed_backbone.get_weights())

    training_model = build_siamese_training_model(embedding_model)
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE_SIAMESE),
        loss=triplet_loss(config.TRIPLET_MARGIN),
    )

    train_gen = TripletGenerator(df)

    history = training_model.fit(
        train_gen,
        epochs=config.EPOCHS_SIAMESE,
        verbose=1,
    )

    embedding_model.save(config.SIAMESE_MODEL_PATH)
    print(f"\nSaved fine-tuned Siamese embedding model -> {config.SIAMESE_MODEL_PATH}")
    return embedding_model, history


def main():
    if not os.path.exists(config.SUBSET_CSV):
        raise FileNotFoundError(
            f"{config.SUBSET_CSV} not found. Run `python src/data_prep.py` first."
        )
    df = pd.read_csv(config.SUBSET_CSV)

    warmed_backbone = stage1_transfer_learning_warmup(df)
    stage2_siamese_triplet_training(df, warmed_backbone)


if __name__ == "__main__":
    main()
