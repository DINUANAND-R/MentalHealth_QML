# ==========================================
# pca.py
# PCA Dimensionality Reduction
# ==========================================

import os
import joblib

from sklearn.decomposition import PCA

from config import PCA_COMPONENTS, MODELS_DIR


def apply_pca(X_train, X_test):

    print("\n" + "=" * 60)
    print("PCA DIMENSIONALITY REDUCTION")
    print("=" * 60)

    # Create PCA object
    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=42
    )

    # Fit and transform training data
    X_train_pca = pca.fit_transform(X_train.toarray())

    # Transform testing data
    X_test_pca = pca.transform(X_test.toarray())

    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save PCA model
    joblib.dump(
        pca,
        os.path.join(MODELS_DIR, "pca.pkl")
    )

    print("\nSaved:")
    print("✓ pca.pkl")

    print(f"\nOriginal Features : {X_train.shape[1]}")
    print(f"Reduced Features  : {PCA_COMPONENTS}")

    print(
        f"\nExplained Variance : "
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
    )

    print("\nPCA Completed Successfully")

    print(f"\nTraining Data Shape after PCA : {X_train_pca.shape}")
    print(f"Testing Data Shape after PCA  : {X_test_pca.shape}")

    return X_train_pca, X_test_pca