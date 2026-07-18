import os
import joblib

from config import MODELS_DIR

# -----------------------------
# Load Saved Models
# -----------------------------
tfidf = joblib.load(
    os.path.join(MODELS_DIR, "tfidf.pkl")
)

pca = joblib.load(
    os.path.join(MODELS_DIR, "pca.pkl")
)

scaler = joblib.load(
    os.path.join(MODELS_DIR, "scaler.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODELS_DIR, "label_encoder.pkl")
)

model = joblib.load(
    os.path.join(MODELS_DIR, "quantum_qsvc.pkl")
)

print("=" * 60)
print("QUANTUM MENTAL HEALTH PREDICTION")
print("=" * 60)

while True:

    text = input("\nEnter your feelings (type 'exit' to quit): ")

    if text.lower() == "exit":
        break

    # -----------------------------
    # TF-IDF
    # -----------------------------
    X = tfidf.transform([text])

    # -----------------------------
    # PCA
    # -----------------------------
    X = pca.transform(X.toarray())

    # -----------------------------
    # Scaling
    # -----------------------------
    X = scaler.transform(X)

    # -----------------------------
    # Quantum Prediction
    # -----------------------------
    prediction = model.predict(X)

    label = label_encoder.inverse_transform(prediction)

    print("\nPrediction :", label[0])