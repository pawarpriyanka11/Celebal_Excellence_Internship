"""
Visual Product Search - Streamlit UI
--------------------------------------
Upload a product photo, get top-K visually similar products.
Features:
  - Toggle between baseline and Siamese model
  - Side-by-side model comparison mode
  - Multiselect category filter
  - Shared embed_single_image utility (no duplication)

Run:
    streamlit run app/streamlit_app.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from siamese_model import L2Normalize  # noqa: F401 - registers custom layer
from feature_extractor import build_backbone, embed_single_image

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import tensorflow as tf


st.set_page_config(page_title="Visual Product Search", layout="wide")


# ----------------------------------------------------------------------
# Cached resource loaders
# ----------------------------------------------------------------------
@st.cache_resource
def load_baseline_model():
    return build_backbone(trainable=False)


@st.cache_resource
def load_siamese_model():
    if not os.path.exists(config.SIAMESE_MODEL_PATH):
        return None
    return tf.keras.models.load_model(config.SIAMESE_MODEL_PATH, compile=False, safe_mode=False)


@st.cache_data
def load_index_data(which: str):
    emb_path = config.BASELINE_EMB_PATH if which == "baseline" else config.SIAMESE_EMB_PATH
    meta_path = config.BASELINE_META_PATH if which == "baseline" else config.SIAMESE_META_PATH
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        return None, None, None
    embeddings = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    # Build a fresh in-memory FAISS index from the index-split rows only,
    # so top_idx from search always aligns with index_meta positions.
    index_mask = (meta["split"] == "index").to_numpy()
    index_embeddings = embeddings[index_mask]
    faiss_index = None
    if FAISS_AVAILABLE:
        dim = index_embeddings.shape[1]
        fi = faiss.IndexFlatIP(dim)
        fi.add(index_embeddings)
        faiss_index = fi
    return embeddings, meta, faiss_index


def _search(faiss_index, embeddings, query_emb, top_k):
    if faiss_index is not None:
        q = query_emb.reshape(1, -1).astype("float32")
        scores, indices = faiss_index.search(q, top_k)
        return indices[0], scores[0]
    from feature_extractor import search_cosine_sklearn
    return search_cosine_sklearn(embeddings, query_emb, top_k=top_k)


def _run_search(which, query_image, top_k, category_filter):
    """Embed + search for one model. Returns (results_df, embed_ms, search_ms)."""
    embeddings, meta, faiss_index = load_index_data(which)
    if embeddings is None:
        return None, 0, 0

    model = load_baseline_model() if which == "baseline" else load_siamese_model()
    if model is None:
        return None, 0, 0

    index_mask = (meta["split"] == "index").to_numpy()
    index_embeddings = embeddings[index_mask]
    index_meta = meta[index_mask].reset_index(drop=True)  # 0-based positions

    t0 = time.time()
    query_emb = embed_single_image(model, query_image)
    embed_ms = (time.time() - t0) * 1000

    t0 = time.time()
    # search against index_embeddings — top_idx are positions into index_embeddings
    # (not the full embeddings array), so iloc on index_meta is safe
    top_idx, scores = _search(faiss_index, index_embeddings, query_emb, top_k * 3)
    search_ms = (time.time() - t0) * 1000

    # clip to valid range defensively (FAISS can return -1 for unfilled slots)
    valid = (top_idx >= 0) & (top_idx < len(index_meta))
    top_idx, scores = top_idx[valid], scores[valid]

    results_df = index_meta.iloc[top_idx].copy()
    results_df["similarity"] = scores

    if category_filter:
        mask = results_df["articleType"].isin(category_filter)
        results_df = results_df[mask]

    return results_df.head(top_k), embed_ms, search_ms


def _render_results(results_df, model_label, embed_ms, search_ms, top_k):
    st.caption(
        f"Model: {model_label}  |  Embed: {embed_ms:.0f} ms  |  Search: {search_ms:.0f} ms"
    )
    if results_df is None or results_df.empty:
        st.warning("No results (embeddings missing or all filtered out).")
        return
    cols_per_row = 5
    for chunk_start in range(0, len(results_df), cols_per_row):
        chunk = results_df.iloc[chunk_start:chunk_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (_, item) in zip(cols, chunk.iterrows()):
            with col:
                if os.path.exists(str(item["image_path"])):
                    st.image(item["image_path"], use_container_width=True)
                st.markdown(f"**{str(item.get('productDisplayName', 'Product'))[:40]}**")
                st.caption(f"{item['articleType']}  ·  sim: {item['similarity']:.3f}")


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🛍️ Visual Product Search")
st.caption(
    "Upload a product photo and get visually similar items — "
    "built with a CNN embedding search engine."
)

with st.sidebar:
    st.header("Settings")

    siamese_available = os.path.exists(config.SIAMESE_MODEL_PATH)
    model_options = ["Baseline (ImageNet CNN)"]
    if siamese_available:
        model_options.append("Siamese (fine-tuned)")

    compare_mode = st.checkbox(
        "Compare models side-by-side",
        value=False,
        disabled=not siamese_available,
        help="Shows baseline and Siamese results next to each other.",
    )

    if not compare_mode:
        model_choice = st.radio("Model", model_options)
    else:
        model_choice = None  # both models used

    top_k = st.slider("Number of results (Top-K)", min_value=3, max_value=20,
                       value=config.TOP_K)

    # Multiselect category filter populated from available categories
    _, meta_for_filter, _ = load_index_data("baseline")
    all_categories = sorted(meta_for_filter["articleType"].unique()) if meta_for_filter is not None else []
    category_filter = st.multiselect(
        "Filter by category", options=all_categories, default=[]
    )

uploaded_file = st.file_uploader(
    "Upload a product image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    query_image = Image.open(uploaded_file)

    col_query, col_results = st.columns([1, 4])
    with col_query:
        st.subheader("Your image")
        st.image(query_image, use_container_width=True)

    with col_results:
        if compare_mode:
            st.subheader(f"Top {top_k} results — side-by-side comparison")
            left, right = st.columns(2)
            with st.spinner("Searching with both models..."):
                base_df, base_emb_ms, base_search_ms = _run_search(
                    "baseline", query_image, top_k, category_filter
                )
                siam_df, siam_emb_ms, siam_search_ms = _run_search(
                    "siamese", query_image, top_k, category_filter
                )
            with left:
                st.markdown("**Baseline (ImageNet CNN)**")
                _render_results(base_df, "Baseline", base_emb_ms, base_search_ms, top_k)
            with right:
                st.markdown("**Siamese (fine-tuned)**")
                _render_results(siam_df, "Siamese", siam_emb_ms, siam_search_ms, top_k)
        else:
            which = "baseline" if model_choice.startswith("Baseline") else "siamese"
            st.subheader(f"Top {top_k} similar products")
            with st.spinner("Finding visually similar products..."):
                results_df, embed_ms, search_ms = _run_search(
                    which, query_image, top_k, category_filter
                )
            _render_results(results_df, model_choice, embed_ms, search_ms, top_k)
else:
    st.info("Upload an image above to see visually similar products.")

