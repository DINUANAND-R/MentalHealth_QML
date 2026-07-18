import joblib
import os

from config import MODELS_DIR


# Load saved objects
tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf.pkl"))
pca = joblib.load(os.path.join(MODELS_DIR, "pca.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))
label_encoder = joblib.load(
    os.path.join(MODELS_DIR, "label_encoder.pkl")
)


def predict(text):

    # TF-IDF
    X = tfidf.transform([text])

    # PCA
    X = pca.transform(X.toarray())

    # Scaling
    X = scaler.transform(X)

    # Prediction
    prediction = model.predict(X)

    # Decode label
    label = label_encoder.inverse_transform(prediction)

    return label[0]