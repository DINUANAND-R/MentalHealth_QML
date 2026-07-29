# ==========================================
# main.py
# Hybrid Quantum Mental Health Assessment
# ==========================================
#
# Pipeline Overview:
#
#  Classical Track:
#    Text → TF-IDF → PCA(8D) → Scaling → Logistic / SVM
#
#  Quantum Track 1 (QSVC):
#    Text → TF-IDF → PCA(8D) → Scaling → QSVC
#
#  Quantum Track 2 (VQC) ← NEW:
#    Text → BERT(768D) → PCA(8D) → Scaling[0,π] → VQC
#
#  Deep Learning Track:
#    Text → Fine-tuned BERT (sequence classification)
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

from train_bert import train_bert

# ── New BERT → VQC pipeline imports ──────────────────────
from bert_embeddings import extract_bert_embeddings
from bert_pca import apply_bert_pca
from bert_scaler import scale_bert_features
from vqc_classifier import train_vqc

from sklearn.model_selection import train_test_split
import numpy as np

from config import (
    BERT_VQC_TRAIN_SIZE,
    BERT_VQC_TEST_SIZE,
    RANDOM_STATE
)


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
    # STEP 11 : PREPARE QUANTUM DATA (QSVC)
    # ==========================================
    print("\nStep 11 : Preparing Quantum Dataset (QSVC)")

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
    # STEP 13 : FINE-TUNE BERT
    # ==========================================
    print("\nStep 13 : Fine-Tuning BERT (Sequence Classification)")

    train_bert(df)

    # ==========================================
    # STEP 14 : BERT EMBEDDINGS EXTRACTION
    # ==========================================
    print("\n" + "=" * 60)
    print("Step 14 : Extracting BERT Embeddings for VQC")
    print("=" * 60)

    # Use a stratified balanced subset for quantum feasibility
    texts  = df["text"].tolist()
    labels = df["status"].values

    # Encode labels using the same encoder as BERT classifier
    import joblib, os
    from config import MODELS_DIR
    bert_encoder = joblib.load(
        os.path.join(MODELS_DIR, "bert_label_encoder.pkl")
    )
    labels_enc = bert_encoder.transform(labels)

    # Stratified split to get train/test subsets
    X_texts_train, X_texts_test, y_vqc_train, y_vqc_test = train_test_split(
        texts,
        labels_enc,
        test_size=0.20,
        stratify=labels_enc,
        random_state=RANDOM_STATE
    )

    # Further subsample for quantum feasibility
    idx_train = []
    idx_test  = []
    unique_classes = np.unique(y_vqc_train)
    per_class_train = BERT_VQC_TRAIN_SIZE // len(unique_classes)
    per_class_test  = BERT_VQC_TEST_SIZE  // len(unique_classes)

    for cls in unique_classes:
        cls_idx_tr = np.where(np.array(y_vqc_train) == cls)[0]
        cls_idx_te = np.where(np.array(y_vqc_test)  == cls)[0]

        chosen_tr = cls_idx_tr[:per_class_train]
        chosen_te = cls_idx_te[:per_class_test]

        idx_train.extend(chosen_tr.tolist())
        idx_test.extend(chosen_te.tolist())

    X_vqc_texts_train = [X_texts_train[i] for i in idx_train]
    X_vqc_texts_test  = [X_texts_test[i]  for i in idx_test]
    y_vqc_train_small = np.array([y_vqc_train[i] for i in idx_train])
    y_vqc_test_small  = np.array([y_vqc_test[i]  for i in idx_test])

    print(f"\n  VQC Train Samples : {len(y_vqc_train_small)}")
    print(f"  VQC Test  Samples : {len(y_vqc_test_small)}")

    # Extract 768-D BERT embeddings
    print("\n  Extracting Training Embeddings...")
    X_bert_train = extract_bert_embeddings(X_vqc_texts_train)

    print("\n  Extracting Test Embeddings...")
    X_bert_test  = extract_bert_embeddings(X_vqc_texts_test)

    # ==========================================
    # STEP 15 : BERT → PCA
    # ==========================================
    print("\nStep 15 : Applying PCA to BERT Embeddings (768D → 8D)")

    X_bert_train_pca, X_bert_test_pca = apply_bert_pca(
        X_bert_train,
        X_bert_test
    )

    # ==========================================
    # STEP 16 : BERT-PCA FEATURE SCALING [0, π]
    # ==========================================
    print("\nStep 16 : Scaling BERT-PCA Features to [0, π]")

    X_bert_train_scaled, X_bert_test_scaled = scale_bert_features(
        X_bert_train_pca,
        X_bert_test_pca
    )

    # ==========================================
    # STEP 17 : VARIATIONAL QUANTUM CLASSIFIER
    # ==========================================
    print("\nStep 17 : Training Variational Quantum Classifier (VQC)")

    vqc_model, vqc_accuracy = train_vqc(
        X_bert_train_scaled,
        X_bert_test_scaled,
        y_vqc_train_small,
        y_vqc_test_small
    )

    # ==========================================
    # FINAL COMPARISON
    # ==========================================
    print("\n")
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(f"{'Model':<35} {'Accuracy':>10}")
    print("-" * 47)
    print(f"{'Logistic Regression (TF-IDF)':<35} {logistic_accuracy:>10.4f}")
    print(f"{'Linear SVM (TF-IDF)':<35} {svm_accuracy:>10.4f}")
    print(f"{'Quantum QSVC (TF-IDF → Quantum)':<35} {quantum_accuracy:>10.4f}")
    print(f"{'VQC (BERT → PCA → Quantum)':<35} {vqc_accuracy:>10.4f}")
    print("=" * 60)

    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    main()