import joblib
import os

from config import MODELS_DIR


# Load saved models
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf.pkl"))
pca = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

# Choose the model you want
#model = joblib.load(os.path.join(MODELS_DIR, "quantum_qsvc.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))
# For classical model instead:
# model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))

print("=" * 60)
print("MENTAL HEALTH PREDICTION")
print("=" * 60)

while True:

    text = input("\nEnter your feelings (type 'exit' to quit): ")

    if text.lower() == "exit":
        break

    # TF-IDF
    X = tfidf.transform([text])

    # PCA
    X = pca.transform(X.toarray())

    # Scaling
    X = scaler.transform(X)

    # Prediction
    pred = model.predict(X)

    label = label_encoder.inverse_transform(pred)[0]

    print("\nPrediction :", label)
    print("Current Working Directory:", os.getcwd())
    print("SVM expects:", model.n_features_in_, "features")