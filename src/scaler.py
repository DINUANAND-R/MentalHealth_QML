# ==========================================
# scaler.py
# Feature Scaling
# ==========================================

import os
import joblib

from sklearn.preprocessing import MinMaxScaler

from config import MODELS_DIR


def scale_features(X_train, X_test):

    print("\n" + "=" * 60)
    print("FEATURE SCALING")
    print("=" * 60)

    # Create MinMax Scaler
    scaler = MinMaxScaler()

    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform testing data
    X_test_scaled = scaler.transform(X_test)

    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save scaler
    joblib.dump(
        scaler,
        os.path.join(MODELS_DIR, "scaler.pkl")
    )

    print("\nSaved:")
    print("✓ scaler.pkl")

    print("\nScaling Completed")
    print("Range : [0,1]")

    print(f"\nTraining Data Shape : {X_train_scaled.shape}")
    print(f"Testing Data Shape  : {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled