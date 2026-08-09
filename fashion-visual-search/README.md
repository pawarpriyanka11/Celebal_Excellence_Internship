# Visual Product Recommendation Engine

An image-based "shop similar" recommendation system — upload a product photo, get back visually similar items with no text search involved.

Pipeline: **EfficientNetB0 embeddings → FAISS similarity search → Siamese network fine-tuning (triplet loss) → Streamlit UI**

Built to run fully on **CPU** — no GPU required.

---

## 1. Setup

```bash
cd fashion-visual-search
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the dataset

Dataset: [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) (Kaggle).

**Option A — Kaggle CLI**
```bash
# requires kaggle.json API token in ~/.kaggle/ (from kaggle.com/settings)
kaggle datasets download -d paramaggarwal/fashion-product-images-dataset -p data/raw --unzip
```

**Option B — Manual download**
Download and unzip from the Kaggle page, then arrange as:
```
data/raw/styles.csv
data/raw/images/<id>.jpg
```

## 3. Run the pipeline (in order)

```bash
# Step 1: Build the subset (11 categories x up to 400 images each)
python src/data_prep.py

# Step 2: Extract EfficientNetB0 (ImageNet) embeddings + build FAISS index
python src/feature_extractor.py

# Step 3: Transfer learning warm-up + Siamese triplet training
python src/train_siamese.py

# Step 4: Build FAISS index from the fine-tuned Siamese model
python src/build_siamese_index.py

# Step 5: Evaluate — Precision@K, Recall@K, MAP@K, timing, qualitative grids
python src/evaluate.py

# Step 6: Launch the UI
streamlit run app/streamlit_app.py
```

Step 5 writes:
- `reports/evaluation_results.csv` — overall metrics
- `reports/evaluation_per_category.csv` — per-category breakdown
- `reports/baseline_qualitative.png` / `reports/siamese_qualitative.png` — visual grids

## 4. Configuration (`src/config.py`)

All paths and hyperparameters are controlled from a single file:

| Parameter | Value | Description |
|---|---|---|
| `BACKBONE` | `"efficientnet"` | EfficientNetB0 (swap to `"mobilenet"` for faster CPU) |
| `IMG_SIZE` | `224` | Input image size (drop to `160` if extraction is slow) |
| `SELECTED_CATEGORIES` | 11 categories | Tshirts, Shirts, Casual Shoes, Sports Shoes, Dresses, Handbags, Watches, Sunglasses, Kurtas, Sandals, Tops |
| `SAMPLES_PER_CATEGORY` | `400` | Max images per category (~4,400 total subset) |
| `QUERY_FRACTION` | `0.15` | Held-out query split for honest evaluation |
| `TOP_K` | `10` | Default number of retrieved results |
| `SIAMESE_EMBED_DIM` | `128` | Projection dimension for Siamese embeddings |
| `TRIPLET_MARGIN` | `0.3` | Triplet loss margin |
| `BATCH_SIZE` | `16` | Small on purpose — CPU only |
| `EPOCHS_FINE_TUNE` | `3` | Stage 1: transfer learning warm-up epochs |
| `EPOCHS_SIAMESE` | `8` | Stage 2: triplet loss training epochs |
| `STEPS_PER_EPOCH` | `80` | Triplet steps per epoch |
| `UNFREEZE_LAST_N_LAYERS` | `20` | Backbone layers unfrozen for fine-tuning |
| `LEARNING_RATE_FINE_TUNE` | `1e-4` | Stage 1 learning rate |
| `LEARNING_RATE_SIAMESE` | `1e-4` | Stage 2 learning rate |

## 5. Model architecture

**Baseline:**
- EfficientNetB0 (ImageNet weights, classification head removed)
- Global Average Pooling → 1280-d L2-normalized embedding
- Retrieval: FAISS `IndexFlatIP` (inner product on L2-normalized = cosine similarity)

**Siamese Network (core enhancement):**
- Same EfficientNetB0 backbone, last 20 layers unfrozen
- Tower: backbone → Dense(256, relu) → BatchNorm → Dropout(0.2) → Dense(128) → L2Normalize
- Training: triplet loss with margin=0.3 on (anchor, positive, negative) image triplets
- Two-stage training: classification warm-up (Stage 1) → triplet loss (Stage 2)

## 6. Evaluation results

| Model | Precision@10 | Recall@10 | Avg Query Time |
|---|---|---|---|
| Baseline | 87.9% | 4.1% | 15.7 ms |
| Siamese | **92.7%** | **4.4%** | **2.2 ms** |

Ground truth relevance uses `articleType` from `styles.csv` — a standard proxy for this dataset since no explicit pairwise similarity labels exist.

Metrics computed on the held-out query split (15% per category, never seen during training).

## 7. Streamlit UI features

- Upload any product image (jpg, png, webp)
- Toggle between Baseline and Siamese model
- Side-by-side model comparison mode
- Category filter (multiselect)
- Displays similarity score and product name per result
- Shows embed time and search time per query

## 8. CPU timing expectations (~4,400-image subset)

| Step | Approx. time |
|---|---|
| `data_prep.py` | 2–5 min |
| `feature_extractor.py` | 5–12 min |
| `train_siamese.py` (3 warm-up + 8 triplet epochs, 80 steps/epoch) | 25–55 min |
| `build_siamese_index.py` | 5–12 min |
| `evaluate.py` | 1–3 min |

To speed up training: lower `IMG_SIZE` to `160`, or reduce `EPOCHS_SIAMESE` / `STEPS_PER_EPOCH` in `src/config.py`.

## 9. Project structure

```
fashion-visual-search/
├── data/
│   ├── raw/                        ← Kaggle dataset (not committed, ~25 GB)
│   └── subset/                     ← generated by data_prep.py (not committed, ~200 MB)
├── src/
│   ├── config.py                   ← all paths & hyperparameters
│   ├── data_prep.py                ← Step 1: subset creation + query split
│   ├── feature_extractor.py        ← Step 2: EfficientNetB0 embeddings + FAISS index
│   ├── siamese_model.py            ← Siamese architecture + triplet loss
│   ├── triplet_data_generator.py   ← on-the-fly (anchor, positive, negative) triplets
│   ├── train_siamese.py            ← Step 3: warm-up + triplet training
│   ├── build_siamese_index.py      ← Step 4: Siamese embeddings + FAISS index
│   └── evaluate.py                 ← Step 5: Precision@K, Recall@K, MAP@K, timing
├── app/
│   └── streamlit_app.py            ← Step 6: upload-an-image UI
├── embeddings/                     ← .npy embeddings + FAISS indexes (generated)
├── reports/                        ← evaluation_results.csv + qualitative PNGs (generated)
├── requirements.txt
└── config.py                       ← root-level config (mirrors src/config.py paths)
```

## 10. Tech stack

- TensorFlow 2.16 — EfficientNetB0 backbone, Siamese network, triplet loss
- FAISS (faiss-cpu) — fast similarity search
- Scikit-learn — cosine similarity fallback
- Streamlit — interactive UI
- Pandas / NumPy — data handling
- Pillow / OpenCV — image processing
- Kaggle CLI — dataset download
