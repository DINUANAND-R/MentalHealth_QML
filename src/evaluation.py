from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate(model, X_test, y_test):

    prediction = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, prediction)

    print("=" * 50)
    print("Accuracy")
    print("=" * 50)
    print(f"{accuracy:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, prediction))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, prediction))

    return accuracy