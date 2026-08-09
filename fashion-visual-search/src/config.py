"""
Central configuration for the Visual Product Recommendation System.
Every script imports paths / hyperparameters from here so the pipeline
stays consistent from data prep -> training -> evaluation -> UI.
"""
import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
STYLES_CSV = os.path.join(RAW_DIR, "styles.csv")
RAW_IMAGES_DIR = os.path.join(RAW_DIR, "images")

SUBSET_DIR = os.path.join(BASE_DIR, "data", "subset")
SUBSET_CSV = os.path.join(SUBSET_DIR, "subset.csv")
SUBSET_IMAGES_DIR = os.path.join(SUBSET_DIR, "images")

EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

BASELINE_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "baseline_embeddings.npy")
BASELINE_FAISS_PATH = os.path.join(EMBEDDINGS_DIR, "baseline.index")
BASELINE_META_PATH = os.path.join(EMBEDDINGS_DIR, "baseline_meta.csv")

SIAMESE_MODEL_PATH = os.path.join(EMBEDDINGS_DIR, "siamese_embedding_model.keras")
SIAMESE_EMB_PATH = os.path.join(EMBEDDINGS_DIR, "siamese_embeddings.npy")
SIAMESE_FAISS_PATH = os.path.join(EMBEDDINGS_DIR, "siamese.index")
SIAMESE_META_PATH = os.path.join(EMBEDDINGS_DIR, "siamese_meta.csv")

# ----------------------------------------------------------------------
# Dataset subsetting strategy
# (Kaggle: paramaggarwal/fashion-product-images-dataset)
# styles.csv columns used: id, gender, masterCategory, subCategory,
#                           articleType, baseColour, season, year,
#                           usage, productDisplayName
# ----------------------------------------------------------------------
SELECTED_CATEGORIES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Dresses",
    "Handbags",
    "Watches",
    "Sunglasses",
    "Kurtas",
    "Sandals",
    "Tops",
]  # expanded to 11 categories for better coverage and Recall@K

SAMPLES_PER_CATEGORY = 400          # increased from 250 for better Recall@K
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------
IMG_SIZE = 224          # drop to 160 in this file if CPU extraction feels slow
CHANNELS = 3

# ----------------------------------------------------------------------
# Feature extraction backbone
# ----------------------------------------------------------------------
# Options: "mobilenet" (MobileNetV2, 1280-d) | "efficientnet" (EfficientNetB0, 1280-d)
# EfficientNetB0 gives better accuracy at the same CPU cost.
BACKBONE = "efficientnet"
BASELINE_EMBED_DIM = 1280   # both MobileNetV2 and EfficientNetB0 GAP output dim

# ----------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------
TOP_K = 10

# ----------------------------------------------------------------------
# Train / query split for evaluation
# (query set is held out to measure Precision@K / Recall@K honestly)
# ----------------------------------------------------------------------
QUERY_FRACTION = 0.15

# ----------------------------------------------------------------------
# Siamese / triplet network
# ----------------------------------------------------------------------
SIAMESE_EMBED_DIM = 128
TRIPLET_MARGIN = 0.3
BATCH_SIZE = 16              # small on purpose - CPU only
EPOCHS_FINE_TUNE = 3         # stage 1: light transfer-learning warmup
EPOCHS_SIAMESE = 8           # stage 2: triplet loss training
STEPS_PER_EPOCH = 80         # triplets per epoch (keeps epochs short on CPU)
LEARNING_RATE_FINE_TUNE = 1e-4
LEARNING_RATE_SIAMESE = 1e-4
UNFREEZE_LAST_N_LAYERS = 20  # how many backbone layers to unfreeze for fine-tuning

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(SUBSET_IMAGES_DIR, exist_ok=True)
