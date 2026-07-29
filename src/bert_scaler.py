# ==========================================
# bert_scaler.py
# Feature Scaling for BERT → VQC Pipeline
# ==========================================
#
# Scales BERT-PCA features to [0, π] range.
#
# Why [0, π]?
#   Quantum rotation gates (Ry, Rz, etc.) use
#   angles in radians. The natural range for
#   angle encoding is [0, π], ensuring full
#   exploration of the Bloch sphere.
#
# Saves bert_scaler.pkl separately from the
# TF-IDF pipeline's scaler.pkl.
# ==========================================

import os
import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from config import MODELS_DIR, BERT_SCALE_MIN, BERT_SCALE_MAX


def scale_bert_features(X_train_pca, X_test_pca):
    """
    Scale BERT-PCA features to [0, π] for quantum angle encoding.

    Parameters
    ----------
    X_train_pca : np.ndarray (n_train, 8)
    X_test_pca  : np.ndarray (n_test,  8)

    Returns
    -------
    X_train_scaled : np.ndarray (n_train, 8) in [0, π]
    X_test_scaled  : np.ndarray (n_test,  8) in [0, π]
    """

    print("\n" + "=" * 60)
    print("BERT → FEATURE SCALING [0, π]")
    print("=" * 60)

    # MinMaxScaler into [BERT_SCALE_MIN, BERT_SCALE_MAX]
    scaler = MinMaxScaler(
        feature_range=(BERT_SCALE_MIN, BERT_SCALE_MAX)
    )

    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled  = scaler.transform(X_test_pca)

    print(f"\n  Train  → min={X_train_scaled.min():.4f}  max={X_train_scaled.max():.4f}")
    print(f"  Test   → min={X_test_scaled.min():.4f}  max={X_test_scaled.max():.4f}")
    print(f"  Range  : [{BERT_SCALE_MIN:.4f}, {BERT_SCALE_MAX:.4f}]  (quantum angle encoding)")

    # ── Save ────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    scaler_path = os.path.join(MODELS_DIR, "bert_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print("\n  Saved:")
    print("  ✓ bert_scaler.pkl")

    return X_train_scaled, X_test_scaled
