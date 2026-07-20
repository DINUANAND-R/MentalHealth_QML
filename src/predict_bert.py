# ==========================================
# predict_bert.py
# ==========================================

import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

from sklearn.preprocessing import LabelEncoder
import pandas as pd

from config import DATASET_PATH


print("=" * 60)
print("BERT MENTAL HEALTH PREDICTION")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)

encoder = LabelEncoder()
encoder.fit(df["status"])

tokenizer = BertTokenizer.from_pretrained(
    "../models/bert_model"
)

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

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    print(
        "\nPrediction :",
        encoder.inverse_transform([prediction])[0]
    )