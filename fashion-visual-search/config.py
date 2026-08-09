"""
Central configuration for the Visual Product Recommendation System.
All paths and hyperparameters used across scripts live here so they
stay consistent between feature extraction, training and the app.
"""
import os

# ---------------------------------------------------------------- paths ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")          # kaggle download lands here
SUBSET_DIR = os.path.join(ROOT_DIR, "data", "subset")         # curated subset
SUBSET_IMAGES_DIR = os.path.join(SUBSET_DIR, "images")
SUBSET_LABELS_CSV = os.path.join(SUBSET_DIR, "labels.csv")

SAMPLE_DATA_DIR = os.path.join(ROOT_DIR, "sample_data")       # synthetic smoke-test set
SAMPLE_LABELS_CSV = os.path.join(SAMPLE_DATA_DIR, "labels.csv")

MODELS_DIR = os.path.join(ROOT_DIR, "models")
EMBEDDINGS_DIR = os.path.join(MODELS_DIR, "embeddings")

# ------------------------------------------------------------- dataset ----
# 5-8 categories as required by the problem statement. Names match the
# `articleType` column of styles.csv in the Kaggle dataset.
DEFAULT_CATEGORIES = [
    "Shirts",
    "Tshirts",
    "Casual Shoes",
    "Sports Shoes",
    "Dresses",
    "Handbags",
    "Watches",
    "Sunglasses",
]
IMAGES_PER_CATEGORY = 250     # ~200-300 per category as specified
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15

# ------------------------------------------------------------ imaging -----
IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ------------------------------------------------------------- models -----
EMBEDDING_DIM = 128            # projection dim used for finetuned / siamese heads
BACKBONE = "resnet50"          # or "efficientnet"

# transfer learning
TL_EPOCHS = 10
TL_BATCH_SIZE = 32
TL_LR = 1e-4
TL_UNFREEZE_LAST_N = 15        # number of backbone layers to unfreeze

# siamese / triplet
SIAMESE_EPOCHS = 20
SIAMESE_BATCH_SIZE = 16
SIAMESE_LR = 1e-4
TRIPLET_MARGIN = 0.3
TRIPLETS_PER_EPOCH = 2000

# retrieval
TOP_K_DEFAULT = 10
USE_FAISS = True

SEED = 42

for _d in [MODELS_DIR, EMBEDDINGS_DIR]:
    os.makedirs(_d, exist_ok=True)
