"""
Step 6: Siamese Network (core enhancement)
--------------------------------------------
Builds a triplet-loss embedding network on top of the same MobileNetV2
backbone used in the baseline, so it can be fine-tuned to place
same-category (visually similar) products closer together in embedding
space than different-category products.

Architecture:
    shared MobileNetV2 backbone (partially unfrozen)
      -> GlobalAveragePooling (already built into backbone, pooling='avg')
      -> Dense(256, relu)
      -> Dense(SIAMESE_EMBED_DIM)
      -> L2 normalize

The same embedding sub-network is applied to anchor / positive / negative
images (shared weights = "Siamese"), and trained with triplet loss.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

import config
from feature_extractor import build_backbone


@tf.keras.utils.register_keras_serializable()
class L2Normalize(layers.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)


def build_embedding_network(unfreeze_last_n=config.UNFREEZE_LAST_N_LAYERS):
    """The tower that will be shared across anchor/positive/negative."""
    backbone = build_backbone(trainable=True)

    # freeze everything, then unfreeze only the last N layers
    # (keeps CPU fine-tuning fast + avoids destroying pretrained features)
    for layer in backbone.layers[:-unfreeze_last_n]:
        layer.trainable = False

    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS))
    x = backbone(inputs)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(config.SIAMESE_EMBED_DIM)(x)
    outputs = L2Normalize(name="l2_normalize")(x)

    embedding_model = Model(inputs, outputs, name="embedding_network")
    return embedding_model


def triplet_loss(margin=config.TRIPLET_MARGIN):
    """Standard triplet loss: max(0, d(a,p) - d(a,n) + margin),
    using squared Euclidean distance on L2-normalized embeddings."""
    def loss_fn(y_true, y_pred):
        # y_pred is the concatenation [anchor_emb, positive_emb, negative_emb]
        # each of width SIAMESE_EMBED_DIM, produced by build_siamese_training_model
        d = config.SIAMESE_EMBED_DIM
        anchor = y_pred[:, 0:d]
        positive = y_pred[:, d:2 * d]
        negative = y_pred[:, 2 * d:3 * d]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss_fn


def build_siamese_training_model(embedding_model):
    """Wraps the shared embedding tower into a 3-input training model
    whose output is [anchor_emb | positive_emb | negative_emb] so that
    triplet_loss can be used directly as a Keras loss."""
    input_shape = (config.IMG_SIZE, config.IMG_SIZE, config.CHANNELS)
    anchor_in = layers.Input(shape=input_shape, name="anchor")
    positive_in = layers.Input(shape=input_shape, name="positive")
    negative_in = layers.Input(shape=input_shape, name="negative")

    anchor_emb = embedding_model(anchor_in)
    positive_emb = embedding_model(positive_in)
    negative_emb = embedding_model(negative_in)

    merged = layers.Concatenate(axis=1)([anchor_emb, positive_emb, negative_emb])
    training_model = Model(
        inputs=[anchor_in, positive_in, negative_in],
        outputs=merged,
        name="siamese_triplet_model",
    )
    return training_model
