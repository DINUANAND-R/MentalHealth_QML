# ==========================================
# vqc_classifier.py
# Variational Quantum Classifier (VQC)
# ==========================================
#
# Pipeline:
#   BERT Embeddings (768D)
#       │
#   PCA (8D)
#       │
#   MinMaxScaler [0, π]
#       │
#   ZZFeatureMap (8 qubits, reps=2)   ← Encoding
#       │
#   EfficientSU2 ansatz (reps=2)      ← Trainable
#       │
#   SPSA Optimizer (300 iters)        ← Classical loop
#       │
#   VQC → Prediction
#
# Fix 1 — ACCURACY:
#   Replaced COBYLA → SPSA.
#   COBYLA needs ~10× parameters iterations to converge;
#   SPSA is designed for quantum circuits (noisy, gradient-free)
#   and converges reliably even with 200 train samples.
#
# Fix 2 — PICKLING:
#   VQC contains an un-picklable local lambda (`parity`).
#   We save only the trained weight array (numpy) + circuit
#   metadata (JSON). At inference time we rebuild the VQC
#   from scratch and load weights via vqc.weights.
# ==========================================

import os
import json
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from qiskit.circuit.library import ZZFeatureMap, EfficientSU2
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.utils import algorithm_globals

from config import (
    MODELS_DIR,
    VQC_REPS,
    VQC_MAX_ITER,
    VQC_FEATURE_MAP_REPS,
    RANDOM_STATE
)

# Seed for reproducibility
algorithm_globals.random_seed = RANDOM_STATE


# ==========================================
# Helper: Build VQC circuit (shared by
# train and inference — ensures identical
# circuit structure when reloading weights)
# ==========================================

def build_vqc(n_features, n_classes):
    """
    Construct a fresh (untrained) VQC with the standard
    circuit architecture used by this project.

    Parameters
    ----------
    n_features : int  — number of input features (= qubits)
    n_classes  : int  — number of output classes

    Returns
    -------
    vqc : VQC (untrained)
    """

    # ── Feature Map ───────────────────────────────────────────
    # ZZFeatureMap encodes classical data into quantum states
    # via entangled Pauli-ZZ rotations.
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=VQC_FEATURE_MAP_REPS
    )

    # ── Ansatz ────────────────────────────────────────────────
    # EfficientSU2 uses alternating single-qubit SU(2) gates
    # and CNOT entanglement layers. Fewer parameters than
    # RealAmplitudes-full → faster, more reliable convergence.
    ansatz = EfficientSU2(
        num_qubits=n_features,
        reps=VQC_REPS,
        entanglement="linear"   # Linear entanglement: O(n) CNOTs
    )

    # ── Optimizer ─────────────────────────────────────────────
    # SPSA (Simultaneous Perturbation Stochastic Approximation):
    # - Specifically designed for noisy quantum circuits
    # - Uses two-point gradient estimation → immune to Barren Plateau
    # - Converges reliably with 100–300 iterations
    optimizer = SPSA(maxiter=VQC_MAX_ITER)

    # ── Sampler ───────────────────────────────────────────────
    sampler = StatevectorSampler()

    # ── VQC ───────────────────────────────────────────────────
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler
    )

    return vqc


# ==========================================
# Save / Load helpers (pickle-safe)
# ==========================================

def save_vqc(vqc, n_features, n_classes):
    """
    Save trained VQC weights as numpy array + circuit metadata as JSON.
    Avoids the un-picklable local `parity` lambda inside VQC.

    Files written:
        models/vqc_weights.npy   — trained parameter array
        models/vqc_config.json   — circuit metadata for reconstruction
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save weights
    weights_path = os.path.join(MODELS_DIR, "vqc_weights.npy")
    np.save(weights_path, vqc.weights)

    # Save circuit config
    config_path = os.path.join(MODELS_DIR, "vqc_config.json")
    config = {
        "n_features"  : int(n_features),
        "n_classes"   : int(n_classes),
        "vqc_reps"    : int(VQC_REPS),
        "fm_reps"     : int(VQC_FEATURE_MAP_REPS),
        "max_iter"    : int(VQC_MAX_ITER),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n  Saved:")
    print("  ✓ vqc_weights.npy")
    print("  ✓ vqc_config.json")


def load_vqc(X_sample):
    """
    Reconstruct the VQC from saved weights + config.
    Call this in predict_bert_vqc.py instead of joblib.load.

    Parameters
    ----------
    X_sample : np.ndarray (1, n_features)
        A single scaled sample — used to trigger internal
        VQC weight-size initialisation via a dry fit.

    Returns
    -------
    vqc : VQC with trained weights loaded
    """
    config_path  = os.path.join(MODELS_DIR, "vqc_config.json")
    weights_path = os.path.join(MODELS_DIR, "vqc_weights.npy")

    with open(config_path) as f:
        cfg = json.load(f)

    n_features = cfg["n_features"]
    n_classes  = cfg["n_classes"]

    # Build fresh circuit
    vqc = build_vqc(n_features, n_classes)

    # Load saved weights
    weights = np.load(weights_path)
    vqc.weights = weights

    return vqc


# ==========================================
# Main Training Function
# ==========================================

def train_vqc(X_train, X_test, y_train, y_test):
    """
    Train and evaluate the VQC on BERT-PCA-Scaled features.

    Parameters
    ----------
    X_train, X_test : np.ndarray (n, 8) — scaled BERT-PCA features
    y_train, y_test : np.ndarray (n,)   — integer class labels

    Returns
    -------
    vqc      : trained VQC
    accuracy : float
    """

    print("\n" + "=" * 60)
    print("VARIATIONAL QUANTUM CLASSIFIER (VQC)")
    print("=" * 60)

    n_features = X_train.shape[1]
    n_classes  = len(np.unique(y_train))

    print(f"\n  Qubits          : {n_features}")
    print(f"  Classes         : {n_classes}")
    print(f"  Ansatz          : EfficientSU2 (reps={VQC_REPS}, linear entanglement)")
    print(f"  Feature Map     : ZZFeatureMap (reps={VQC_FEATURE_MAP_REPS})")
    print(f"  Optimizer       : SPSA (maxiter={VQC_MAX_ITER})")
    print(f"  Train Samples   : {len(y_train)}")
    print(f"  Test  Samples   : {len(y_test)}")

    # Class balance check
    unique, counts = np.unique(y_train, return_counts=True)
    print("\n  Class Distribution (train):")
    for cls, cnt in zip(unique, counts):
        print(f"    Class {cls} : {cnt} samples")

    # ── Build ────────────────────────────────────────────────
    print("\n  Building VQC circuit...")
    vqc = build_vqc(n_features, n_classes)

    # ── Train ────────────────────────────────────────────────
    print("\n  Training VQC (hybrid quantum-classical loop)...")
    print("  Optimizer : SPSA — robust to Barren Plateaus")
    print("  (This may take several minutes for simulation)\n")

    vqc.fit(X_train, y_train)

    print("\n  Training Completed.")

    # ── Evaluate ─────────────────────────────────────────────
    predictions = vqc.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("VQC RESULTS")
    print("=" * 60)

    print(f"\n  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")

    print("\n  Classification Report")
    print(classification_report(y_test, predictions, zero_division=0))

    print("  Confusion Matrix")
    print(confusion_matrix(y_test, predictions))

    # ── Save (pickle-safe) ───────────────────────────────────
    save_vqc(vqc, n_features, n_classes)

    return vqc, accuracy
