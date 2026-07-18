# scaler.py

import os
import joblib

from sklearn.preprocessing import MinMaxScaler
from config import MODELS_DIR


def scale_features(X_train, X_test):

    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(
        scaler,
        os.path.join(MODELS_DIR, "scaler.pkl")
    )

    print("\nScaling Completed")
    print("Range : [0,1]")

    return X_train_scaled, X_test_scaled