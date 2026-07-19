# ==========================================
# train_logistic.py
# ==========================================

import os
import joblib

from sklearn.linear_model import LogisticRegression
from config import MODELS_DIR


def train_logistic(X_train, y_train):

    print("\n" + "=" * 60)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 60)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(
        model,
        os.path.join(MODELS_DIR, "logistic.pkl")
    )

    print("\nLogistic Regression Model Saved")

    return model