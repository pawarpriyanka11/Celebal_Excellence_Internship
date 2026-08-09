"""
Generates (anchor, positive, negative) image triplets on the fly:
  - anchor & positive: same articleType category (visually similar)
  - negative: hard negative — different category but embedding-closest
    to the anchor (semi-hard mining every HARD_NEG_REFRESH_STEPS steps)

Data augmentation (random flip + brightness jitter) is applied to each
image to improve generalization on the small ~2,000-image subset.
"""
import numpy as np
import tensorflow as tf

import config
from feature_extractor import load_and_preprocess

HARD_NEG_REFRESH_STEPS = 20   # re-mine hard negatives every N steps


def _augment(arr: np.ndarray) -> np.ndarray:
    """Random horizontal flip + brightness jitter on a single HWC float32 array."""
    t = tf.constant(arr)
    t = tf.image.random_flip_left_right(t)
    t = tf.image.random_brightness(t, max_delta=0.1)
    return t.numpy()


class TripletGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=config.BATCH_SIZE,
                 steps_per_epoch=config.STEPS_PER_EPOCH, seed=config.RANDOM_SEED):
        self.df = df[df["split"] == "index"].reset_index(drop=True)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._embeddings = None   # populated lazily for hard negative mining

        self.by_category = {
            cat: self.df[self.df["articleType"] == cat].index.to_numpy()
            for cat in self.df["articleType"].unique()
        }
        self.categories = list(self.by_category.keys())

    # ------------------------------------------------------------------
    # Hard negative mining helpers
    # ------------------------------------------------------------------
    def _refresh_embeddings(self):
        """Compute cheap L2-normalized pixel-mean embeddings for hard
        negative mining (avoids a full forward pass during data loading)."""
        vecs = []
        for _, row in self.df.iterrows():
            arr = load_and_preprocess(row["image_path"])  # (H,W,3)
            vecs.append(arr.mean(axis=(0, 1)))            # (3,) colour mean
        self._embeddings = np.stack(vecs).astype("float32")  # (N, 3)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self._embeddings /= norms

    def _hard_negative_idx(self, anchor_idx: int, neg_pool: np.ndarray) -> int:
        """Return the index in neg_pool whose embedding is closest to anchor."""
        if self._embeddings is None:
            return int(self.rng.choice(neg_pool))
        anchor_emb = self._embeddings[anchor_idx]          # (3,)
        neg_embs = self._embeddings[neg_pool]              # (M, 3)
        sims = neg_embs @ anchor_emb
        return int(neg_pool[np.argmax(sims)])

    # ------------------------------------------------------------------

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        # Refresh hard-negative embeddings periodically
        if self._step % HARD_NEG_REFRESH_STEPS == 0:
            self._refresh_embeddings()
        self._step += 1

        anchors, positives, negatives = [], [], []

        for _ in range(self.batch_size):
            pos_cat = self.rng.choice(self.categories)
            neg_cat = self.rng.choice([c for c in self.categories if c != pos_cat])

            pos_pool = self.by_category[pos_cat]
            neg_pool = self.by_category[neg_cat]

            if len(pos_pool) < 2:
                continue

            a_idx, p_idx = self.rng.choice(pos_pool, size=2, replace=False)
            n_idx = self._hard_negative_idx(int(a_idx), neg_pool)

            anchors.append(_augment(load_and_preprocess(self.df.loc[a_idx, "image_path"])))
            positives.append(_augment(load_and_preprocess(self.df.loc[p_idx, "image_path"])))
            negatives.append(_augment(load_and_preprocess(self.df.loc[n_idx, "image_path"])))

        anchors = np.stack(anchors)
        positives = np.stack(positives)
        negatives = np.stack(negatives)

        dummy_y = np.zeros((len(anchors), 3 * config.SIAMESE_EMBED_DIM), dtype="float32")
        return (anchors, positives, negatives), dummy_y

    def on_epoch_end(self):
        pass
