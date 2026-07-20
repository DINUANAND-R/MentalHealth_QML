# ==========================================
# predict_bert.py
# ==========================================

import joblib
import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

print("=" * 60)
print("BERT MENTAL HEALTH PREDICTION")
print("=" * 60)

# -----------------------------
# Load Label Encoder
# -----------------------------
encoder = joblib.load("../models/bert_label_encoder.pkl")

# -----------------------------
# Load Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained(
    "../models/bert_model"
)

# -----------------------------
# Load Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "../models/bert_model"
)

model.eval()

while True:

    text = input(
        "\nEnter your feelings (type 'exit' to quit): "
    )

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    print("\nPrediction :", encoder.inverse_transform([prediction])[0])

    print("\nConfidence Scores")

    for label, score in zip(
        encoder.classes_,
        probabilities[0]
    ):
        print(f"{label:12s}: {score.item()*100:.2f}%")