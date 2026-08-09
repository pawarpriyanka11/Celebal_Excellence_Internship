"""
Similarity computation and top-K retrieval over precomputed embeddings.

Two backends:
  - cosine (sklearn) : exact, simple, fine for a few thousand items
  - faiss             : IndexFlatIP over L2-normalized vectors == cosine similarity,
                         scales to much larger catalogs; falls back to cosine if
                         faiss isn't installed.
"""
import os
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_K_DEFAULT  # noqa: E402

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class SimilaritySearchIndex:
    """Wraps a set of embeddings + metadata and supports top-K similarity search."""

    def __init__(self, embeddings: np.ndarray, image_paths, categories, use_faiss: bool = True):
        self.embeddings = embeddings.astype("float32")
        self.image_paths = np.asarray(image_paths)
        self.categories = np.asarray(categories)
        self.use_faiss = use_faiss and _HAS_FAISS

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # inner product == cosine on normalized vecs
            self.index.add(self.embeddings)
        else:
            self.index = None

    @classmethod
    def from_npz(cls, npz_path: str, use_faiss: bool = True):
        data = np.load(npz_path, allow_pickle=True)
        return cls(data["embeddings"], data["image_paths"], data["categories"], use_faiss=use_faiss)

    def query(self, query_embedding: np.ndarray, top_k: int = TOP_K_DEFAULT, exclude_self_index: int = None):
        """Return (indices, scores, paths, categories) for the top_k most similar items."""
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        # normalize query too, in case caller forgot
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        k = top_k + (1 if exclude_self_index is not None else 0)
        k = min(k, len(self.embeddings))

        if self.use_faiss:
            scores, idx = self.index.search(query_embedding, k)
            scores, idx = scores[0], idx[0]
        else:
            sims = cosine_similarity(query_embedding, self.embeddings)[0]
            idx = np.argsort(-sims)[:k]
            scores = sims[idx]

        if exclude_self_index is not None:
            mask = idx != exclude_self_index
            idx, scores = idx[mask][:top_k], scores[mask][:top_k]

        return idx, scores, self.image_paths[idx], self.categories[idx]


def cosine_topk(query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = TOP_K_DEFAULT):
    """Stateless convenience function (no FAISS): returns (indices, scores)."""
    sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
    idx = np.argsort(-sims)[:top_k]
    return idx, sims[idx]
