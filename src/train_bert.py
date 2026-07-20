# ==========================================
# train_bert.py
# ==========================================

import os

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    # Training Arguments
    # (Compatible with Transformers 5.x)
    # -----------------------------------------
    training_args = TrainingArguments(

        output_dir="./bert_results",

        do_train=True,
        do_eval=True,

        num_train_epochs=3,

        learning_rate=2e-5,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_steps=100,

        load_best_model_at_end=True,

        report_to="none"
    )

    # -----------------------------------------
    # Trainer
    # -----------------------------------------
    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=test_dataset
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