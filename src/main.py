# ==========================================
# main.py
# Hybrid Quantum Mental Health Assessment
# ==========================================

from data_loader import load_dataset
from eda import perform_eda
from tfidf import build_tfidf
from split_data import split_dataset

from pca import apply_pca
from scaler import scale_features

from train_logistic import train_logistic
from train_svm import train_svm
from evaluation import evaluate

from prepare_quantum_data import prepare_quantum_data
from quantum_qsvc import train_qsvc


def main():

    # ==========================================
    # STEP 1 : LOAD DATASET
    # ==========================================
    print("\nStep 1 : Loading Dataset")

    df = load_dataset()

    # ==========================================
    # STEP 2 : EDA
    # ==========================================
    print("\nStep 2 : Running EDA")

    perform_eda(df)

    # ==========================================
    # STEP 3 : TF-IDF
    # ==========================================
    print("\nStep 3 : Building TF-IDF")

    X, y = build_tfidf(df)

    # ==========================================
    # STEP 4 : TRAIN TEST SPLIT
    # ==========================================
    print("\nStep 4 : Splitting Dataset")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    # ==========================================
    # STEP 5 : PCA
    # ==========================================
    print("\nStep 5 : Applying PCA")

    X_train_pca, X_test_pca = apply_pca(
        X_train,
        X_test
    )

    # ==========================================
    # STEP 6 : FEATURE SCALING
    # ==========================================
    print("\nStep 6 : Scaling Features")

    X_train_scaled, X_test_scaled = scale_features(
        X_train_pca,
        X_test_pca
    )

    # ==========================================
    # STEP 7 : LOGISTIC REGRESSION
    # ==========================================
    print("\nStep 7 : Training Logistic Regression")

    logistic_model = train_logistic(
        X_train_scaled,
        y_train
    )

    print("\nStep 8 : Evaluating Logistic Regression")

    logistic_accuracy = evaluate(
        logistic_model,
        X_test_scaled,
        y_test
    )

    # ==========================================
    # STEP 9 : LINEAR SVM
    # ==========================================
    print("\nStep 9 : Training Linear SVM")

    svm_model = train_svm(
        X_train_scaled,
        y_train
    )

    print("\nStep 10 : Evaluating Linear SVM")

    svm_accuracy = evaluate(
        svm_model,
        X_test_scaled,
        y_test
    )

    # ==========================================
    # STEP 11 : PREPARE QUANTUM DATA
    # ==========================================
    print("\nStep 11 : Preparing Quantum Dataset")

    (
        X_train_quantum,
        X_test_quantum,
        y_train_quantum,
        y_test_quantum
    ) = prepare_quantum_data(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )

    # ==========================================
    # STEP 12 : QUANTUM QSVC
    # ==========================================
    print("\nStep 12 : Training Quantum QSVC")

    quantum_model, quantum_accuracy = train_qsvc(
        X_train_quantum,
        X_test_quantum,
        y_train_quantum,
        y_test_quantum
    )

    # ==========================================
    # FINAL COMPARISON
    # ==========================================
    print("\n")
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(f"Logistic Regression Accuracy : {logistic_accuracy:.4f}")
    print(f"Linear SVM Accuracy          : {svm_accuracy:.4f}")
    print(f"Quantum QSVC Accuracy        : {quantum_accuracy:.4f}")

    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    main()