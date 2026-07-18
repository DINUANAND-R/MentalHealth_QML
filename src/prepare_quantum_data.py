# ==========================================
# prepare_quantum_data.py
# ==========================================

from sklearn.model_selection import train_test_split
from config import QUANTUM_TRAIN_SIZE, QUANTUM_TEST_SIZE


def prepare_quantum_data(
        X_train,
        X_test,
        y_train,
        y_test,
        train_size=QUANTUM_TRAIN_SIZE,
        test_size=QUANTUM_TEST_SIZE
):

    print("\n" + "=" * 60)
    print("PREPARING QUANTUM DATASET")
    print("=" * 60)

    # Balanced training subset
    X_train_small, _, y_train_small, _ = train_test_split(
        X_train,
        y_train,
        train_size=train_size,
        stratify=y_train,
        random_state=42
    )

    # Balanced testing subset
    X_test_small, _, y_test_small, _ = train_test_split(
        X_test,
        y_test,
        train_size=test_size,
        stratify=y_test,
        random_state=42
    )

    print(f"\nQuantum Training Samples : {len(y_train_small)}")
    print(f"Quantum Testing Samples  : {len(y_test_small)}")

    return (
        X_train_small,
        X_test_small,
        y_train_small,
        y_test_small
    )