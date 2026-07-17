import joblib

from sklearn.svm import LinearSVC


def train_svm(X_train, y_train):

    model = LinearSVC()

    model.fit(X_train, y_train)

    joblib.dump(model, "../models/svm.pkl")

    return model