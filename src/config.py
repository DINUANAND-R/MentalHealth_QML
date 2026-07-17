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