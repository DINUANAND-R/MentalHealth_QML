# ==========================================
# tfidf.py
# TF-IDF Feature Extraction Module
# ==========================================

import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from config import (
    TEXT_COLUMN,
    LABEL_COLUMN,
    TFIDF_FEATURES,
    MODELS_DIR
)


def build_tfidf(df):
    """
    Convert text into TF-IDF features
    Encode class labels
    Save TF-IDF vectorizer and Label Encoder
    """

    print("\n" + "=" * 60)
    print("TF-IDF FEATURE EXTRACTION")
    print("=" * 60)

    # -------------------------------
    # Encode Labels
    # -------------------------------

    encoder = LabelEncoder()

    y = encoder.fit_transform(df[LABEL_COLUMN])

    print("\nClasses Found:")

    for index, label in enumerate(encoder.classes_):
        print(f"{index} -> {label}")

    # -------------------------------
    # Build TF-IDF
    # -------------------------------

    tfidf = TfidfVectorizer(
        max_features=TFIDF_FEATURES,
        stop_words="english"
    )

    X = tfidf.fit_transform(df[TEXT_COLUMN])

    print("\nTF-IDF Matrix Shape:")
    print(X.shape)

    print("\nVocabulary Size:")
    print(len(tfidf.vocabulary_))

    # -------------------------------
    # Create models directory
    # -------------------------------

    os.makedirs(MODELS_DIR, exist_ok=True)

    # -------------------------------
    # Save Models
    # -------------------------------

    joblib.dump(
        tfidf,
        os.path.join(MODELS_DIR, "tfidf.pkl")
    )

    joblib.dump(
        encoder,
        os.path.join(MODELS_DIR, "label_encoder.pkl")
    )

    print("\nSaved:")
    print("✓ tfidf.pkl")
    print("✓ label_encoder.pkl")

    return X, y