# ==========================================
# main.py
# Hybrid Quantum Mental Health Assessment
# ==========================================

from data_loader import load_dataset
from eda import perform_eda
from tfidf import build_tfidf
from split_data import split_dataset
from train_svm import train_svm
from evaluation import evaluate
from pca import apply_pca
from scaler import scale_features
from quantum_qsvc import train_qsvc
from prepare_quantum_data import prepare_quantum_data

def main():

    # -----------------------------------------
    # Step 1 : Load Dataset
    # -----------------------------------------
    print("\nStep 1 : Loading Dataset")
    df = load_dataset()

    # -----------------------------------------
    # Step 2 : Exploratory Data Analysis
    # -----------------------------------------
    print("\nStep 2 : Running EDA")
    perform_eda(df)

    # -----------------------------------------
    # Step 3 : TF-IDF Feature Extraction
    # -----------------------------------------
    print("\nStep 3 : Building TF-IDF")
    X, y = build_tfidf(df)

    # -----------------------------------------
    # Step 4 : Train-Test Split
    # -----------------------------------------
    print("\nStep 4 : Splitting Dataset")
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # -----------------------------------------
    # Step 5 : Classical SVM
    # -----------------------------------------
    print("\nStep 5 : Training Classical SVM")
    svm_model = train_svm(X_train, y_train)

    # -----------------------------------------
    # Step 6 : Evaluate Classical SVM
    # -----------------------------------------
    print("\nStep 6 : Evaluating Classical SVM")
    evaluate(svm_model, X_test, y_test)

    # -----------------------------------------
    # Step 7 : PCA
    # -----------------------------------------
    print("\nStep 7 : Applying PCA")

    X_train_pca, X_test_pca = apply_pca(
        X_train,
        X_test
    )

    print("\nPCA Completed Successfully")

    print("\nTraining Data Shape after PCA :", X_train_pca.shape)
    print("Testing Data Shape after PCA  :", X_test_pca.shape)

    print("\nAll Classical Pipeline Steps Completed Successfully")

    print("\nStep 8 : Scaling Features")

    X_train_scaled, X_test_scaled = scale_features(
    X_train_pca,
    X_test_pca
    )

    print("\nStep 9 : Preparing Quantum Dataset")

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
    print("\nStep 10 : Training Quantum QSVC")

    train_qsvc(
    X_train_quantum,
    X_test_quantum,
    y_train_quantum,
    y_test_quantum
    )


if __name__ == "__main__":
    main()