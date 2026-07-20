# ==========================================
# bert_dataset.py
# ==========================================

import torch
from torch.utils.data import Dataset


class MentalHealthDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=128):

        # Works with both pandas Series and Python lists
        if hasattr(texts, "tolist"):
            self.texts = texts.tolist()
        else:
            self.texts = list(texts)

        if hasattr(labels, "tolist"):
            self.labels = labels.tolist()
        else:
            self.labels = list(labels)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }