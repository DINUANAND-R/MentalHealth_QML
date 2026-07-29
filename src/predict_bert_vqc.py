# ==========================================
# predict_bert_vqc.py
# BERT → PCA → Scaling → VQC Prediction
# ==========================================
#
# End-to-end interactive inference.
#
# Pipeline:
#   User Input
#       │
#   BERT [CLS] Embedding (768D)
#       │
#   PCA (8D)           [bert_pca.pkl]
#       │
#   Scaling [0, π]     [bert_scaler.pkl]
#       │
#   VQC → Label        [vqc_weights.npy + vqc_config.json]
# ==========================================

import os
import joblib
import torch
import numpy as np

from transformers import BertTokenizer, BertModel

from config import MODELS_DIR

# Import the pickle-safe loader from vqc_classifier
from vqc_classifier import load_vqc

# ==========================================
# Load Saved Models
# ==========================================

print("=" * 60)
print("BERT → VQC MENTAL HEALTH PREDICTION")
print("=" * 60)

print("\nLoading models...")

# ── Label Encoder ────────────────────────────────────────────
label_encoder = joblib.load(
    os.path.join(MODELS_DIR, "bert_label_encoder.pkl")
)

# ── PCA ──────────────────────────────────────────────────────
bert_pca = joblib.load(
    os.path.join(MODELS_DIR, "bert_pca.pkl")
)

# ── Scaler ───────────────────────────────────────────────────
bert_scaler = joblib.load(
    os.path.join(MODELS_DIR, "bert_scaler.pkl")
)

# ── BERT ─────────────────────────────────────────────────────
_bert_model_path = os.path.join(MODELS_DIR, "bert_model")

if os.path.exists(_bert_model_path):
    print("  Using fine-tuned BERT")
    _bert_path = _bert_model_path
else:
    print("  Using bert-base-uncased (fine-tuned model not found)")
    _bert_path = "bert-base-uncased"

tokenizer  = BertTokenizer.from_pretrained(_bert_path)
bert_model = BertModel.from_pretrained(_bert_path)
bert_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model.to(device)

print(f"  Device : {device.upper()}")

# ── VQC ──────────────────────────────────────────────────────
# Use the pickle-safe loader that reconstructs the circuit
# from vqc_weights.npy + vqc_config.json (no joblib.load)
print("  Loading VQC weights...")

# Provide a dummy sample so load_vqc knows the input shape
_dummy = np.zeros((1, bert_pca.n_components_))
vqc_model = load_vqc(_dummy)

print("  VQC loaded successfully!")
print("\nAll models loaded!\n")


# ==========================================
# Inference Function
# ==========================================

def predict(text: str) -> str:
    """
    Run the full BERT → PCA → Scaler → VQC pipeline on one text.
    """

    # Step 1: BERT embedding ──────────────────────────────────
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs   = bert_model(**encoded)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # shape: (1, 768)

    # Step 2: PCA ─────────────────────────────────────────────
    pca_features = bert_pca.transform(embedding)
    # shape: (1, 8)

    # Step 3: Scaling ─────────────────────────────────────────
    scaled_features = bert_scaler.transform(pca_features)
    # shape: (1, 8)  in [0, pi]

    # Step 4: VQC prediction ──────────────────────────────────
    prediction_idx = vqc_model.predict(scaled_features)

    label = label_encoder.inverse_transform(prediction_idx)

    return label[0]


# ==========================================
# Interactive Loop
# ==========================================

print("=" * 60)
print("Interactive Mental Health Assessment")
print("Using: BERT -> PCA -> Scaler -> VQC")
print("=" * 60)

while True:

    print()
    text = input("Enter your feelings (type 'exit' to quit): ").strip()

    if not text:
        print("  Please enter some text.")
        continue

    if text.lower() == "exit":
        print("\nExiting. Take care!")
        break

    print("\n  Processing through quantum pipeline...")

    try:
        result = predict(text)
        print(f"\n  Prediction : {result}")

    except Exception as e:
        print(f"\n  Error: {e}")
