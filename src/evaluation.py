from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate(model, X_test, y_test):

    prediction = model.predict(X_test)

    print("="*50)
    print("Accuracy")
    print("="*50)

    print(accuracy_score(y_test, prediction))

    print("\nClassification Report")

    print(classification_report(y_test, prediction))

    print("\nConfusion Matrix")

    print(confusion_matrix(y_test, prediction))