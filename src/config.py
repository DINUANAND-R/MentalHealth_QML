import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "mental_heath_feature_engineered.csv"
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

TEXT_COLUMN = "text"

LABEL_COLUMN = "status"

TFIDF_FEATURES = 3000

TEST_SIZE = 0.20

RANDOM_STATE = 42

PCA_COMPONENTS = 8

# ==========================================
# QSVC (TF-IDF Pipeline)
# ==========================================
QUANTUM_TRAIN_SIZE = 200

QUANTUM_TEST_SIZE = 80

# ==========================================
# VQC (BERT Embedding Pipeline)
# ==========================================

# Number of samples for BERT → VQC training
BERT_VQC_TRAIN_SIZE = 200

# Number of samples for BERT → VQC testing
BERT_VQC_TEST_SIZE = 80

# PCA components for BERT embeddings (must match BERT → VQC qubit count)
BERT_PCA_COMPONENTS = 8

# VQC circuit repetitions for RealAmplitudes/EfficientSU2 ansatz
VQC_REPS = 2

# Maximum iterations for classical optimizer (SPSA)
VQC_MAX_ITER = 300

# Feature map repetitions (ZZFeatureMap)
VQC_FEATURE_MAP_REPS = 2

# Scaling range for quantum angle encoding [0, pi]
import math
BERT_SCALE_MIN = 0.0
BERT_SCALE_MAX = math.pi