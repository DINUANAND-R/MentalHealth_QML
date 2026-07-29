# ==========================================
# vqc_classifier.py
# Variational Quantum Classifier (VQC)
# ==========================================
#
# Implements the BERT → VQC pipeline:
#
#   BERT Embeddings (768D)
#       │
#   PCA (8D)
#       │
#   MinMaxScaler [0, π]
#       │
#   ZZFeatureMap (8 qubits, reps=2)  ← Encoding
#       │
#   RealAmplitudes ansatz (reps=VQC_REPS) ← Trainable
#       │
#   COBYLA / SPSA Optimizer           ← Classical loop
#       │
#   VQC → Prediction
#
# VQC differs from QSVC:
#   - QSVC uses a quantum kernel (kernel trick)
#   - VQC uses a parameterized circuit trained
#     via a classical optimizer (hybrid loop)
# ==========================================

import os
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

from config import MODELS_DIR, VQC_REPS, VQC_MAX_ITER, RANDOM_STATE

# Seed for reproducibility
algorithm_globals.random_seed = RANDOM_STATE


def train_vqc(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train a Variational Quantum Classifier on BERT embeddings.

    Parameters
    ----------
    X_train, X_test : np.ndarray (n_samples, 8) — scaled BERT-PCA features
    y_train, y_test : np.ndarray (n_samples,)   — integer class labels

    Returns
    -------
    vqc     : trained VQC model
    accuracy: float
    """

    print("\n" + "=" * 60)
    print("VARIATIONAL QUANTUM CLASSIFIER (VQC)")
    print("=" * 60)

    n_features = X_train.shape[1]
    n_classes  = len(np.unique(y_train))

    print(f"\n  Qubits       : {n_features}")
    print(f"  Classes      : {n_classes}")
    print(f"  Ansatz Reps  : {VQC_REPS}")
    print(f"  Max Iter     : {VQC_MAX_ITER}")
    print(f"  Train Samples: {len(y_train)}")
    print(f"  Test  Samples: {len(y_test)}")

    # ── Feature Map ─────────────────────────────────────
    print("\n  Building ZZFeatureMap...")

    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=2
    )

    # ── Ansatz ──────────────────────────────────────────
    print("  Building RealAmplitudes Ansatz...")

    ansatz = RealAmplitudes(
        num_qubits=n_features,
        reps=VQC_REPS,
        entanglement="full"  # All-to-all entanglement
    )

    # ── Optimizer ───────────────────────────────────────
    print("  Building COBYLA Optimizer...")

    optimizer = COBYLA(maxiter=VQC_MAX_ITER)

    # ── Primitives ──────────────────────────────────────────────
    sampler = StatevectorSampler()

    # ── VQC Model ───────────────────────────────────────
    print("  Building VQC Model...")

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )

    # ── Training ────────────────────────────────────────
    print("\n  Training VQC (hybrid quantum-classical loop)...")
    print("  (This may take several minutes for simulation)")

    vqc.fit(X_train, y_train)

    print("\n  Training Completed.")

    # ── Evaluation ──────────────────────────────────────
    predictions = vqc.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("VQC RESULTS")
    print("=" * 60)

    print(f"\n  Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")

    print("\n  Classification Report")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("  Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    # ── Save ────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "vqc_model.pkl")
    joblib.dump(vqc, model_path)

    print("\n  Saved:")
    print("  ✓ vqc_model.pkl")

    return vqc, accuracy
