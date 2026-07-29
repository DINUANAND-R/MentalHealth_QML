# ==========================================
# bert_embeddings.py
# BERT Feature Extractor (768-D CLS Token)
# ==========================================
#
# This module extracts [CLS] token embeddings
# from BERT for use as features in downstream
# quantum classifiers (VQC).
#
# Pipeline:
#   Text → BERT → 768-D embedding → PCA → VQC
# ==========================================

import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import BertTokenizer, BertModel

from config import MODELS_DIR


def get_bert_model_path():
    """
    Return fine-tuned BERT path if it exists,
    otherwise fall back to pretrained bert-base-uncased.
    """
    fine_tuned_path = os.path.join(MODELS_DIR, "bert_model")

    if os.path.exists(fine_tuned_path):
        print(f"  ✓ Using fine-tuned BERT from: {fine_tuned_path}")
        return fine_tuned_path
    else:
        print("  ℹ Fine-tuned BERT not found. Using bert-base-uncased.")
        return "bert-base-uncased"


def extract_bert_embeddings(
    texts,
    batch_size=32,
    max_length=128,
    device=None
):
    """
    Extract [CLS] token embeddings from BERT.

    Parameters
    ----------
    texts      : list or pd.Series of raw text strings
    batch_size : number of texts processed at once
    max_length : max token length for truncation
    device     : 'cuda' | 'cpu' | None (auto-detect)

    Returns
    -------
    embeddings : np.ndarray of shape (n_samples, 768)
    """

    # ── Device setup ────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Device : {device.upper()}")

    # ── Convert to list ─────────────────────────────────
    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    texts = [str(t) for t in texts]

    # ── Load model ──────────────────────────────────────
    model_path = get_bert_model_path()

    tokenizer = BertTokenizer.from_pretrained(model_path)

    # BertModel gives raw hidden states (no classification head)
    model = BertModel.from_pretrained(model_path)
    model.eval()
    model.to(device)

    # ── Extract in batches ──────────────────────────────
    all_embeddings = []

    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="  Extracting embeddings"):

            batch = texts[i * batch_size : (i + 1) * batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Move to device
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)

            # [CLS] token is the first token → shape (batch, 768)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # Move to CPU numpy
            all_embeddings.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(all_embeddings)

    print(f"\n  Embeddings Shape : {embeddings.shape}")
    print(f"  Dtype            : {embeddings.dtype}")

    return embeddings
