# ==========================================
# quantum_qsvc.py
# ==========================================

import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from qiskit_algorithms.state_fidelities import ComputeUncompute

from config import MODELS_DIR


def train_qsvc(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n" + "=" * 60)
    print("QUANTUM SUPPORT VECTOR CLASSIFIER")
    print("=" * 60)

    print("\nCreating Quantum Feature Map...")

    feature_map = ZZFeatureMap(
        feature_dimension=X_train.shape[1],
        reps=2
    )

    print("Creating Statevector Sampler...")

    sampler = StatevectorSampler()

    print("Creating Fidelity...")

    fidelity = ComputeUncompute(
        sampler=sampler
    )

    print("Creating Quantum Kernel...")

    quantum_kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity
    )

    print("Building QSVC Model...")

    model = QSVC(
        quantum_kernel=quantum_kernel
    )

    print("\nTraining Quantum Model...")

    model.fit(X_train, y_train)

    print("Training Completed.")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nQuantum Accuracy")
    print("=" * 30)
    print(f"{accuracy:.4f}")

    print("\nClassification Report")
    print(classification_report(
        y_test,
        predictions
    ))

    print("\nConfusion Matrix")
    print(confusion_matrix(
        y_test,
        predictions
    ))

    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(
        model,
        os.path.join(
            MODELS_DIR,
            "quantum_qsvc.pkl"
        )
    )

    print("\nQuantum Model Saved")

    return model