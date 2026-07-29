# Project Report
# Hybrid Quantum Mental Health Assessment System

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objectives](#2-objectives)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Pipeline Tracks](#5-pipeline-tracks)
   - [Track 1 — Classical (Logistic Regression & SVM)](#track-1--classical)
   - [Track 2 — Quantum QSVC](#track-2--quantum-qsvc)
   - [Track 3 — BERT Fine-Tuning (Deep Learning)](#track-3--bert-fine-tuning)
   - [Track 4 — BERT → PCA → VQC (New)](#track-4--bert--pca--vqc)
6. [Module Reference](#6-module-reference)
7. [Hyperparameters & Configuration](#7-hyperparameters--configuration)
8. [Saved Model Artifacts](#8-saved-model-artifacts)
9. [How to Run](#9-how-to-run)
10. [Technology Stack](#10-technology-stack)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Future Directions](#12-future-directions)

---

## 1. Project Overview

This project implements a **Hybrid Quantum-Classical Mental Health Assessment System** that classifies mental health conditions from natural language text. It combines four distinct machine learning paradigms — classical ML, deep learning, quantum kernel methods, and variational quantum computing — to benchmark their performance on the same mental health dataset.

The system is designed for **research and comparative analysis** between classical and quantum approaches to NLP-based mental health classification.

---

## 2. Objectives

| # | Objective |
|---|---|
| 1 | Build a robust text classification pipeline for mental health status prediction |
| 2 | Implement classical ML baselines (Logistic Regression, SVM) |
| 3 | Fine-tune BERT for deep learning-based classification |
| 4 | Apply Quantum SVM (QSVC) using a quantum kernel |
| 5 | Implement the new **BERT → PCA → VQC** pipeline |
| 6 | Compare all four approaches in a unified final report |

---

## 3. Dataset

| Property | Detail |
|---|---|
| **File** | `data/mental_heath_feature_engineered.csv` |
| **Text Column** | `text` — raw natural language utterances |
| **Label Column** | `status` — mental health category |
| **Classes** | 4 classes (multi-class classification) |
| **Split** | 80% Train / 20% Test (stratified) |
| **Size** | ~28 MB (feature-engineered CSV) |

The dataset contains free-form text entries describing personal mental health experiences. The `status` column holds categorical labels representing different mental health conditions.

---

## 4. System Architecture

```
                         RAW TEXT INPUT
                               │
           ┌───────────────────┼───────────────────────┐
           │                   │                       │
    TF-IDF (3000)        BERT Fine-Tune          BERT Embeddings
    Vectorizer           (Seq. Classifier)       [CLS] 768D
           │                   │                       │
      PCA (8D)            Training                PCA (8D)
           │             Evaluation                    │
    MinMaxScaler               │              MinMaxScaler [0, π]
    [0, 1]                     │                       │
           │                   │                       │
     ┌─────┴─────┐             │               ┌───────┴──────┐
     │           │             │               │              │
 Logistic     Linear       Fine-tuned        QSVC           VQC
 Regression   SVM (L)        BERT          (Quantum       (Quantum
                                            Kernel)       Variational)
     │           │             │               │              │
     └─────┬─────┘             │               └───────┬──────┘
           │                   │                       │
        Accuracy            Accuracy                Accuracy
```

---

## 5. Pipeline Tracks

### Track 1 — Classical

**Text → TF-IDF → PCA (8D) → MinMaxScaler [0,1] → Logistic Regression / SVM**

This is the baseline classical pipeline shared by two models.

#### Step-by-Step

| Step | Module | Description |
|---|---|---|
| 1 | `data_loader.py` | Reads CSV into a Pandas DataFrame |
| 2 | `eda.py` | Prints dataset statistics; saves class distribution chart |
| 3 | `tfidf.py` | Builds TF-IDF matrix (3000 features), encodes labels with `LabelEncoder` |
| 4 | `split_data.py` | Stratified 80/20 train-test split |
| 5 | `pca.py` | PCA reduces TF-IDF features from 3000 → 8 dimensions |
| 6 | `scaler.py` | MinMaxScaler normalises to [0, 1] |
| 7 | `train_logistic.py` | Trains `LogisticRegression(max_iter=1000)` |
| 8 | `train_svm.py` | Trains `LinearSVC()` |
| 9 | `evaluation.py` | Reports accuracy, classification report, confusion matrix |

#### Why TF-IDF + PCA?
TF-IDF captures statistical word importance across the corpus. PCA to 8 dimensions is required to make the data quantum-compatible (8 features = 8 qubits). The same preprocessing is shared between classical models and the QSVC.

---

### Track 2 — Quantum QSVC

**Text → TF-IDF → PCA (8D) → MinMaxScaler [0,1] → QSVC**

Uses the same preprocessed features as Track 1 but replaces the classical classifier with a **Quantum Support Vector Classifier**.

#### How QSVC Works

| Component | Detail |
|---|---|
| **Feature Map** | `ZZFeatureMap` (8 qubits, reps=3) |
| **Kernel** | `FidelityQuantumKernel` — computes inner products in Hilbert space |
| **Fidelity** | `ComputeUncompute` using `StatevectorSampler` |
| **Classifier** | `QSVC` with `C=10` |
| **Training Size** | 200 samples (quantum simulation is expensive) |
| **Test Size** | 80 samples |

#### QSVC vs Classical SVM
A classical SVM uses dot products in feature space as the kernel. The QSVC replaces this with **quantum fidelity** — the overlap between quantum states prepared from two data points. This allows the kernel to implicitly exploit an exponentially large Hilbert space.

```
Data Point A ──► ZZFeatureMap ──► |ψ_A⟩
                                        }── ⟨ψ_A|ψ_B⟩ = Kernel Value
Data Point B ──► ZZFeatureMap ──► |ψ_B⟩
```

#### Improvement Applied
- Increased `reps` 2 → **3** (deeper entanglement, richer feature space)
- Added `C=10` (stronger regularization margin)

---

### Track 3 — BERT Fine-Tuning

**Text → Tokenizer → BertForSequenceClassification → Prediction**

Fine-tunes a pre-trained `bert-base-uncased` model for direct 4-class mental health classification.

#### Architecture

```
Input Text
    │
BertTokenizer (max_length=128, padding, truncation)
    │
BERT Transformer (12 layers, 768 hidden, 12 heads)
    │
[CLS] Token → Linear Head (768 → 4 classes)
    │
Softmax → Predicted Class
```

#### Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| Epochs | **5** (was 3) | More training → better convergence |
| Learning Rate | `2e-5` | Standard for BERT fine-tuning |
| Warmup Ratio | `0.1` | Prevents large gradient updates at start |
| Weight Decay | `0.01` | L2 regularization, reduces overfitting |
| Batch Size | 16 per device | |
| Gradient Accumulation | 2 steps | Effective batch size = 32 |
| Best Model Loading | ✓ | Keeps best checkpoint by accuracy |
| Metric | accuracy + weighted F1 | Per-epoch tracking via `compute_metrics` |

#### Saved Artifacts
- `models/bert_model/` — fine-tuned weights + tokenizer
- `models/bert_label_encoder.pkl` — shared with VQC pipeline

---

### Track 4 — BERT → PCA → VQC

**Text → BERT [CLS] Embedding (768D) → PCA (8D) → Scaling [0,π] → VQC**

This is the **new pipeline** added to the project. It combines BERT's deep language representations with a **Variational Quantum Classifier** (VQC) — a fundamentally different quantum algorithm from QSVC.

#### Step-by-Step

| Step | Module | Description |
|---|---|---|
| 14 | `bert_embeddings.py` | Extracts 768-D `[CLS]` embeddings from BERT |
| 15 | `bert_pca.py` | PCA: 768D → 8D (saved as `bert_pca.pkl`) |
| 16 | `bert_scaler.py` | MinMaxScaler to **[0, π]** (saved as `bert_scaler.pkl`) |
| 17 | `vqc_classifier.py` | Trains VQC, evaluates, saves `vqc_model.pkl` |

#### BERT as Feature Extractor
In Track 3, BERT is a full classifier. In Track 4, BERT is used as a **frozen feature extractor** — only the `[CLS]` token representation (768D) is taken as input to the quantum model. This removes the fine-tuned classification head and feeds richer semantic features into the quantum circuit.

```
BERT Layer 12
    │
[CLS] hidden state → 768-D vector
    │                (rich semantic representation)
    ▼
PCA (8D)
```

#### Why [0, π] Scaling?
Quantum rotation gates (Ry, Rz) accept angles in radians. Scaling to [0, π] ensures features map directly to meaningful quantum state rotations on the Bloch sphere, covering the full upper hemisphere.

```
|0⟩ ──── Ry(θ) ───► cos(θ/2)|0⟩ + sin(θ/2)|1⟩
                     θ ∈ [0, π] → full range of superpositions
```

#### VQC Architecture

```
Input (8 features, scaled to [0, π])
    │
ZZFeatureMap (8 qubits, reps=2) ← Data Encoding Layer
    │
RealAmplitudes Ansatz (reps=3, full entanglement) ← Trainable Layer
    │
Measurement → Class Probabilities
    │
COBYLA Optimizer (max_iter=150) ← Classical Feedback Loop
    │
Trained VQC → Prediction
```

#### VQC vs QSVC — Key Difference

| Property | QSVC | VQC |
|---|---|---|
| Type | Kernel method | Variational (hybrid) |
| Training | Kernel matrix computation | Iterative parameter optimization |
| Parameters | None (kernel-based) | Trainable circuit parameters |
| Optimizer | Implicit (SVM dual) | COBYLA classical optimizer |
| Analogy | Quantum Kernel SVM | Quantum Neural Network |
| Strengths | Proven convergence | More flexible, near-term friendly |

#### VQC Training Dataset Preparation
Since quantum simulation is computationally expensive, a **balanced stratified subset** is used:
- Per class: 50 train samples + 20 test samples
- Total: **200 train / 80 test** (balanced across all 4 classes)

---

## 6. Module Reference

| Module | Track | Role |
|---|---|---|
| [config.py](file:///e:/Research/src/config.py) | All | Central configuration: paths, hyperparameters, constants |
| [data_loader.py](file:///e:/Research/src/data_loader.py) | All | CSV loader |
| [eda.py](file:///e:/Research/src/eda.py) | All | Exploratory data analysis + class distribution plot |
| [tfidf.py](file:///e:/Research/src/tfidf.py) | 1, 2 | TF-IDF vectorizer + label encoder |
| [split_data.py](file:///e:/Research/src/split_data.py) | 1, 2 | Stratified train/test split |
| [pca.py](file:///e:/Research/src/pca.py) | 1, 2 | PCA on TF-IDF features → 8D |
| [scaler.py](file:///e:/Research/src/scaler.py) | 1, 2 | MinMaxScaler [0, 1] |
| [train_logistic.py](file:///e:/Research/src/train_logistic.py) | 1 | Logistic Regression |
| [train_svm.py](file:///e:/Research/src/train_svm.py) | 1 | Linear SVM |
| [evaluation.py](file:///e:/Research/src/evaluation.py) | 1, 2 | Accuracy + classification report + confusion matrix |
| [prepare_quantum_data.py](file:///e:/Research/src/prepare_quantum_data.py) | 2 | Balanced quantum subset (200/80) |
| [quantum_qsvc.py](file:///e:/Research/src/quantum_qsvc.py) | 2 | ZZFeatureMap + FidelityKernel + QSVC |
| [train_bert.py](file:///e:/Research/src/train_bert.py) | 3 | BERT fine-tuning (5 epochs, warmup, weight decay) |
| [bert_dataset.py](file:///e:/Research/src/bert_dataset.py) | 3, 4 | PyTorch Dataset for BERT tokenization |
| [bert_embeddings.py](file:///e:/Research/src/bert_embeddings.py) | 4 | [CLS] embedding extractor (768D) |
| [bert_pca.py](file:///e:/Research/src/bert_pca.py) | 4 | PCA on BERT embeddings → 8D |
| [bert_scaler.py](file:///e:/Research/src/bert_scaler.py) | 4 | MinMaxScaler [0, π] |
| [vqc_classifier.py](file:///e:/Research/src/vqc_classifier.py) | 4 | ZZFeatureMap + RealAmplitudes + COBYLA → VQC |
| [main.py](file:///e:/Research/src/main.py) | All | Orchestrates all 17 steps end-to-end |
| [predict_quantum.py](file:///e:/Research/src/predict_quantum.py) | 2 | Interactive QSVC inference |
| [predict_bert.py](file:///e:/Research/src/predict_bert.py) | 3 | Interactive BERT inference |
| [predict_bert_vqc.py](file:///e:/Research/src/predict_bert_vqc.py) | 4 | Interactive BERT → VQC inference |

---

## 7. Hyperparameters & Configuration

All configurable parameters are centralised in [config.py](file:///e:/Research/src/config.py):

### Dataset
| Parameter | Value | Description |
|---|---|---|
| `TEXT_COLUMN` | `"text"` | Input text column name |
| `LABEL_COLUMN` | `"status"` | Target label column name |
| `TEST_SIZE` | `0.20` | 20% held-out test set |
| `RANDOM_STATE` | `42` | Reproducibility seed |

### Classical Pipeline
| Parameter | Value | Description |
|---|---|---|
| `TFIDF_FEATURES` | `3000` | Vocabulary size for TF-IDF |
| `PCA_COMPONENTS` | `8` | TF-IDF PCA target dimensions |

### QSVC
| Parameter | Value | Description |
|---|---|---|
| `QUANTUM_TRAIN_SIZE` | `200` | Training samples for quantum simulation |
| `QUANTUM_TEST_SIZE` | `80` | Test samples for quantum simulation |
| ZZFeatureMap `reps` | `3` | Entanglement depth |
| QSVC `C` | `10` | SVM regularization strength |

### BERT Fine-Tuning
| Parameter | Value | Description |
|---|---|---|
| Epochs | `5` | Training epochs |
| Learning Rate | `2e-5` | Standard BERT LR |
| `warmup_ratio` | `0.1` | 10% warmup steps |
| `weight_decay` | `0.01` | L2 regularization |
| `gradient_accumulation_steps` | `2` | Effective batch = 32 |

### VQC (BERT → Quantum)
| Parameter | Value | Description |
|---|---|---|
| `BERT_VQC_TRAIN_SIZE` | `200` | VQC training samples |
| `BERT_VQC_TEST_SIZE` | `80` | VQC test samples |
| `BERT_PCA_COMPONENTS` | `8` | BERT PCA target dimensions |
| `VQC_REPS` | `3` | RealAmplitudes ansatz depth |
| `VQC_MAX_ITER` | `150` | COBYLA optimizer iterations |
| `BERT_SCALE_MIN/MAX` | `[0.0, π]` | Quantum angle encoding range |

---

## 8. Saved Model Artifacts

After running `main.py`, the `models/` directory contains:

| File | Size | Description |
|---|---|---|
| `tfidf.pkl` | ~109 KB | TF-IDF vectorizer (3000 features) |
| `label_encoder.pkl` | ~516 B | Label encoder for TF-IDF pipeline |
| `pca.pkl` | ~217 KB | PCA fitted on TF-IDF features |
| `scaler.pkl` | ~983 B | MinMaxScaler [0,1] for TF-IDF |
| `logistic.pkl` | ~1.2 KB | Trained Logistic Regression |
| `svm.pkl` | ~1.0 KB | Trained Linear SVM |
| `quantum_qsvc.pkl` | ~75 KB | Trained QSVC model |
| `bert_label_encoder.pkl` | ~516 B | Label encoder shared by BERT + VQC |
| `bert_model/` | ~400 MB | Fine-tuned BERT weights + tokenizer |
| `bert_pca.pkl` | — | PCA fitted on BERT embeddings (new) |
| `bert_scaler.pkl` | — | MinMaxScaler [0, π] for BERT (new) |
| `vqc_model.pkl` | — | Trained VQC model (new) |

---

## 9. How to Run

### Full Pipeline (All 4 Tracks)
```powershell
cd e:\Research\src
python main.py
```

This runs all **17 steps** sequentially and prints a final comparison table:

```
Model                               Accuracy
-----------------------------------------------
Logistic Regression (TF-IDF)          x.xxxx
Linear SVM (TF-IDF)                   x.xxxx
Quantum QSVC (TF-IDF → Quantum)       x.xxxx
VQC (BERT → PCA → Quantum)            x.xxxx
```

### Interactive Inference Scripts

```powershell
# Classical QSVC — quantum kernel inference
python predict_quantum.py

# Fine-tuned BERT — deep learning inference
python predict_bert.py

# BERT → VQC — new quantum pipeline inference
python predict_bert_vqc.py
```

### Expected Runtime (approximate)

| Step | Expected Time |
|---|---|
| TF-IDF + PCA + Classical Models | < 1 minute |
| QSVC (200 samples, 8 qubits, reps=3) | 10–30 minutes |
| BERT Fine-Tuning (5 epochs) | 1–3 hours (GPU) / longer (CPU) |
| BERT Embedding Extraction | 5–15 minutes |
| VQC Training (200 samples, 8 qubits) | 30–90 minutes |

> **Note:** The quantum steps (QSVC and VQC) use statevector simulation which is computationally intensive. A GPU is strongly recommended for BERT fine-tuning.

---

## 10. Technology Stack

| Category | Library | Version |
|---|---|---|
| **Core** | NumPy, Pandas, SciPy | ≥2.0 |
| **Classical ML** | scikit-learn | ≥1.7 |
| **Deep Learning** | PyTorch | ≥2.7 |
| **Transformers** | Hugging Face `transformers` | 5.14.x |
| **Quantum** | Qiskit | 2.5.0 |
| **Quantum ML** | qiskit-machine-learning | 0.9.0 |
| **Quantum Algorithms** | qiskit-algorithms | ≥0.4 |
| **Visualization** | Matplotlib, Seaborn | ≥3.10 |
| **Serialization** | joblib | ≥1.5 |
| **NLP** | NLTK, spaCy, TextBlob | — |

---

## 11. Key Design Decisions

### 1. Why PCA to 8 dimensions?
Quantum simulation complexity scales **exponentially** with the number of qubits. 8 qubits is a practical upper bound for statevector simulation on a laptop/workstation. 8 PCA components also capture the dominant variance in both TF-IDF and BERT feature spaces.

### 2. Why separate PCA/Scaler for BERT?
The TF-IDF and BERT features live in completely different spaces:
- TF-IDF: sparse, high-dimensional, statistical
- BERT: dense, semantic, contextual

Using the same PCA fitted on TF-IDF for BERT features would produce meaningless transformations. Separate `bert_pca.pkl` and `bert_scaler.pkl` files prevent this.

### 3. Why [0, π] scaling for VQC instead of [0, 1]?
Quantum rotation gates encode data as rotation **angles** in radians. The full range [0, π] covers the complete upper hemisphere of the Bloch sphere (all superposition states). Scaling to [0, 1] would only cover a fraction of meaningful quantum states.

### 4. Why COBYLA for VQC optimizer?
COBYLA (Constrained Optimization BY Linear Approximations) is a **gradient-free** optimizer — it does not require computing parameter gradients. This is important because:
- Quantum circuits can suffer from the **Barren Plateau** problem (vanishing gradients)
- Gradient-free methods are more stable for small training sets
- COBYLA has good empirical performance on NISQ-era VQC problems

### 5. Why `RealAmplitudes` ansatz with full entanglement?
`RealAmplitudes` with `entanglement="full"` creates all-to-all connections between qubits, giving the model maximum expressive power for 8 qubits. This is important because the BERT-PCA features may have complex correlations that require global entanglement to capture.

### 6. Why use quantum models on small datasets?
Current quantum hardware and simulators cannot compete with classical models on large datasets. The quantum models here serve a **research purpose** — to benchmark the feasibility and relative performance of quantum approaches on NLP tasks, using small balanced subsets that are computationally tractable.

---

## 12. Future Directions

| Direction | Description |
|---|---|
| **Noise Simulation** | Run on `qiskit-aer` with noise models to simulate real hardware |
| **More Qubits** | Try 12–16 qubits with reduced dataset for richer representation |
| **SPSA Optimizer** | Replace COBYLA with SPSA for better gradient-free convergence |
| **Quantum Transfer Learning** | Fine-tune only the ansatz while keeping feature map fixed |
| **Real Hardware** | Deploy on IBM Quantum hardware via `QiskitRuntimeService` |
| **Streamlit Dashboard** | Build an interactive web app (already in requirements) |
| **Larger VQC Dataset** | Use GPU-accelerated simulation (qiskit-aer GPU) for more samples |
| **Ensemble** | Combine BERT + VQC predictions for a hybrid ensemble classifier |

---

*Report generated: 2026-07-29 | Project: MentalHealth_QML | Author: Research*
