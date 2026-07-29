import os
import numpy as np

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

from bert_dataset import MentalHealthDataset
from config import MODELS_DIR


def train_bert(df):

    print("\n" + "=" * 60)
    print("BERT TRAINING")
    print("=" * 60)

    # -----------------------------------------
    # Tokenizer
    # -----------------------------------------
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    # -----------------------------------------
    # Label Encoding
    # -----------------------------------------
    encoder = LabelEncoder()

    labels = encoder.fit_transform(df["status"])

    # Save label encoder
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(
        encoder,
        os.path.join(MODELS_DIR, "bert_label_encoder.pkl")
    )

    # -----------------------------------------
    # Train Test Split
    # -----------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # -----------------------------------------
    # Dataset
    # -----------------------------------------
    train_dataset = MentalHealthDataset(
        X_train.tolist(),
        y_train,
        tokenizer
    )

    test_dataset = MentalHealthDataset(
        X_test.tolist(),
        y_test,
        tokenizer
    )

    # -----------------------------------------
    # Model
    # -----------------------------------------
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=4
    )

    # -----------------------------------------
    # Compute Metrics
    # -----------------------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1  = f1_score(labels, predictions, average="weighted", zero_division=0)
        return {"accuracy": acc, "f1": f1}

    # -----------------------------------------
    # Training Arguments
    # -----------------------------------------
    training_args = TrainingArguments(

        output_dir="./bert_results",

        do_train=True,
        do_eval=True,

        num_train_epochs=5,           # Increased from 3 → 5

        learning_rate=2e-5,

        warmup_ratio=0.1,             # 10% warmup steps

        weight_decay=0.01,            # L2 regularization

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        gradient_accumulation_steps=2, # Effective batch size = 32

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_steps=50,

        load_best_model_at_end=True,

        metric_for_best_model="accuracy",

        greater_is_better=True,

        report_to="none"
    )

    # -----------------------------------------
    # Trainer
    # -----------------------------------------
    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=test_dataset,

        compute_metrics=compute_metrics  # Per-epoch accuracy tracking
    )

    # -----------------------------------------
    # Train
    # -----------------------------------------
    trainer.train()

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------
    results = trainer.evaluate()

    print("\nEvaluation Results")
    print(results)

    # -----------------------------------------
    # Save Model
    # -----------------------------------------
    model.save_pretrained(
        os.path.join(MODELS_DIR, "bert_model")
    )

    tokenizer.save_pretrained(
        os.path.join(MODELS_DIR, "bert_model")
    )

    print("\nSaved:")
    print("✓ bert_model")
    print("✓ bert_label_encoder.pkl")

    return model