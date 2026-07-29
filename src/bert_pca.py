# ==========================================
# bert_pca.py
# PCA on BERT Embeddings (768D → 8D)
# ==========================================
#
# Applies PCA dimensionality reduction to
# BERT CLS embeddings before quantum encoding.
#
# Saves a SEPARATE PCA model from the
# TF-IDF pipeline (bert_pca.pkl).
# ==========================================

import os
import joblib
import numpy as np

from sklearn.decomposition import PCA

from config import BERT_PCA_COMPONENTS, MODELS_DIR


def apply_bert_pca(X_train_emb, X_test_emb):
    """
    Fit PCA on BERT training embeddings and transform both sets.

    Parameters
    ----------
    X_train_emb : np.ndarray (n_train, 768) — BERT embeddings
    X_test_emb  : np.ndarray (n_test,  768) — BERT embeddings

    Returns
    -------
    X_train_pca : np.ndarray (n_train, BERT_PCA_COMPONENTS)
    X_test_pca  : np.ndarray (n_test,  BERT_PCA_COMPONENTS)
    """

    print("\n" + "=" * 60)
    print("BERT → PCA DIMENSIONALITY REDUCTION")
    print("=" * 60)

    print(f"\n  Input  Shape : {X_train_emb.shape}")
    print(f"  Target Dims  : {BERT_PCA_COMPONENTS}")

    # ── Fit PCA ─────────────────────────────────────────
    pca = PCA(
        n_components=BERT_PCA_COMPONENTS,
        random_state=42
    )

    X_train_pca = pca.fit_transform(X_train_emb)
    X_test_pca  = pca.transform(X_test_emb)

    explained = pca.explained_variance_ratio_.sum() * 100

    print(f"\n  Explained Variance : {explained:.2f}%")
    print(f"  Train Shape after  : {X_train_pca.shape}")
    print(f"  Test  Shape after  : {X_test_pca.shape}")

    # ── Save ────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    pca_path = os.path.join(MODELS_DIR, "bert_pca.pkl")
    joblib.dump(pca, pca_path)

    print("\n  Saved:")
    print("  ✓ bert_pca.pkl")

    return X_train_pca, X_test_pca
