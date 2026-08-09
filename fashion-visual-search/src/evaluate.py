"""
Step 7: Evaluation
---------------------
Compares baseline (frozen CNN) vs Siamese fine-tuned embeddings using:
  - Precision@K, Recall@K, MAP   (ground truth = same articleType)
  - Per-category breakdown
  - Inference time per query
  - A saved visual grid (query + top-K) per model

Uses FAISS for retrieval when available (falls back to sklearn cosine).

Run (after both baseline and siamese indexes have been built):
    python src/evaluate.py
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import config
from feature_extractor import search_cosine_sklearn

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def precision_recall_at_k(retrieved_categories, query_category, total_relevant, k):
    retrieved_k = retrieved_categories[:k]
    relevant_in_k = sum(1 for c in retrieved_k if c == query_category)
    precision = relevant_in_k / k
    recall = relevant_in_k / total_relevant if total_relevant > 0 else 0.0
    return precision, recall


def average_precision(retrieved_categories, query_category, k):
    """AP = mean of precision@i for each i where result i is relevant."""
    hits, running_precision = 0, 0.0
    for i, cat in enumerate(retrieved_categories[:k], start=1):
        if cat == query_category:
            hits += 1
            running_precision += hits / i
    return running_precision / hits if hits > 0 else 0.0


# ----------------------------------------------------------------------
# Retrieval (FAISS preferred, sklearn fallback)
# ----------------------------------------------------------------------
def _load_faiss_index(index_path):
    if FAISS_AVAILABLE and os.path.exists(index_path):
        return faiss.read_index(index_path)
    return None


def _search(faiss_index, index_embeddings, query_emb, k):
    if faiss_index is not None:
        from feature_extractor import search_faiss
        return search_faiss(faiss_index, query_emb, top_k=k)
    return search_cosine_sklearn(index_embeddings, query_emb, top_k=k)


# ----------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------
def evaluate_model(embeddings, meta_df, model_name, k=config.TOP_K, faiss_index_path=None):
    index_mask = (meta_df["split"] == "index").to_numpy()
    query_mask = (meta_df["split"] == "query").to_numpy()

    index_embeddings = embeddings[index_mask]
    index_categories = meta_df.loc[index_mask, "articleType"].to_numpy()
    query_positions = np.where(query_mask)[0]
    category_counts = pd.Series(index_categories).value_counts().to_dict()

    faiss_idx = _load_faiss_index(faiss_index_path) if faiss_index_path else None
    retrieval_method = "FAISS" if faiss_idx is not None else "sklearn-cosine"
    print(f"  Retrieval method: {retrieval_method}")

    precisions, recalls, aps, query_times = [], [], [], []
    per_cat = {cat: {"precisions": [], "recalls": [], "aps": []} for cat in np.unique(index_categories)}

    for qpos in query_positions:
        query_emb = embeddings[qpos]
        query_cat = meta_df.loc[qpos, "articleType"]
        total_relevant = category_counts.get(query_cat, 0)

        t0 = time.time()
        top_idx, _ = _search(faiss_idx, index_embeddings, query_emb, k)
        query_times.append(time.time() - t0)

        retrieved_cats = index_categories[top_idx]
        p, r = precision_recall_at_k(retrieved_cats, query_cat, total_relevant, k)
        ap = average_precision(retrieved_cats, query_cat, k)

        precisions.append(p)
        recalls.append(r)
        aps.append(ap)
        if query_cat in per_cat:
            per_cat[query_cat]["precisions"].append(p)
            per_cat[query_cat]["recalls"].append(r)
            per_cat[query_cat]["aps"].append(ap)

    # Overall results
    results = {
        "model": model_name,
        "retrieval": retrieval_method,
        "num_queries": len(query_positions),
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"MAP@{k}": float(np.mean(aps)),
        "avg_query_time_ms": float(np.mean(query_times) * 1000),
    }

    # Per-category breakdown
    cat_rows = []
    for cat, vals in per_cat.items():
        if vals["precisions"]:
            cat_rows.append({
                "model": model_name,
                "category": cat,
                f"precision@{k}": float(np.mean(vals["precisions"])),
                f"recall@{k}": float(np.mean(vals["recalls"])),
                f"MAP@{k}": float(np.mean(vals["aps"])),
            })
    per_cat_df = pd.DataFrame(cat_rows).sort_values("category")

    return results, per_cat_df


# ----------------------------------------------------------------------
# Qualitative grid
# ----------------------------------------------------------------------
def save_qualitative_grid(embeddings, meta_df, model_name, k=5, n_examples=4,
                          faiss_index_path=None):
    index_mask = (meta_df["split"] == "index").to_numpy()
    query_positions = np.where((meta_df["split"] == "query").to_numpy())[0]
    example_positions = np.random.default_rng(config.RANDOM_SEED).choice(
        query_positions, size=min(n_examples, len(query_positions)), replace=False
    )

    index_embeddings = embeddings[index_mask]
    index_paths = meta_df.loc[index_mask, "image_path"].to_numpy()
    faiss_idx = _load_faiss_index(faiss_index_path) if faiss_index_path else None

    fig, axes = plt.subplots(len(example_positions), k + 1,
                              figsize=(2.2 * (k + 1), 2.4 * len(example_positions)))
    if len(example_positions) == 1:
        axes = axes[np.newaxis, :]

    for row, qpos in enumerate(example_positions):
        query_emb = embeddings[qpos]
        top_idx, _ = _search(faiss_idx, index_embeddings, query_emb, k)

        axes[row, 0].imshow(Image.open(meta_df.loc[qpos, "image_path"]))
        axes[row, 0].set_title("QUERY", fontsize=9)
        axes[row, 0].axis("off")
        for col, ridx in enumerate(top_idx):
            axes[row, col + 1].imshow(Image.open(index_paths[ridx]))
            axes[row, col + 1].axis("off")

    fig.suptitle(f"{model_name}: query vs top-{k} retrieved", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(config.REPORTS_DIR, f"{model_name}_qualitative.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved qualitative grid -> {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    all_results, all_per_cat = [], []

    if os.path.exists(config.BASELINE_EMB_PATH):
        emb = np.load(config.BASELINE_EMB_PATH)
        meta = pd.read_csv(config.BASELINE_META_PATH)
        print("Evaluating baseline embeddings ...")
        res, per_cat = evaluate_model(emb, meta, "baseline",
                                      faiss_index_path=config.BASELINE_FAISS_PATH)
        all_results.append(res)
        all_per_cat.append(per_cat)
        save_qualitative_grid(emb, meta, "baseline",
                              faiss_index_path=config.BASELINE_FAISS_PATH)
    else:
        print(f"[skip] {config.BASELINE_EMB_PATH} not found.")

    if os.path.exists(config.SIAMESE_EMB_PATH):
        emb = np.load(config.SIAMESE_EMB_PATH)
        meta = pd.read_csv(config.SIAMESE_META_PATH)
        print("Evaluating Siamese fine-tuned embeddings ...")
        res, per_cat = evaluate_model(emb, meta, "siamese",
                                      faiss_index_path=config.SIAMESE_FAISS_PATH)
        all_results.append(res)
        all_per_cat.append(per_cat)
        save_qualitative_grid(emb, meta, "siamese",
                              faiss_index_path=config.SIAMESE_FAISS_PATH)
    else:
        print(f"[skip] {config.SIAMESE_EMB_PATH} not found.")

    if all_results:
        results_df = pd.DataFrame(all_results)
        per_cat_df = pd.concat(all_per_cat, ignore_index=True)

        out_csv = os.path.join(config.REPORTS_DIR, "evaluation_results.csv")
        per_cat_csv = os.path.join(config.REPORTS_DIR, "evaluation_per_category.csv")
        results_df.to_csv(out_csv, index=False)
        per_cat_df.to_csv(per_cat_csv, index=False)

        print(f"\n=== Overall Results ===\n{results_df.to_string(index=False)}")
        print(f"\n=== Per-Category Results ===\n{per_cat_df.to_string(index=False)}")
        print(f"\nSaved -> {out_csv}")
        print(f"Saved -> {per_cat_csv}")
    else:
        print("No embeddings found to evaluate.")


if __name__ == "__main__":
    main()
