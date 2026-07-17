from data_loader import load_dataset
from eda import perform_eda
from tfidf import build_tfidf
from split_data import split_dataset
from train_svm import train_svm
from evaluation import evaluate


def main():

    print("Step 1: Loading Dataset")
    df = load_dataset()

    print("Step 2: Running EDA")
    perform_eda(df)

    print("Step 3: Building TF-IDF")
    X, y = build_tfidf(df)

    print("Step 4: Splitting Dataset")
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("Step 5: Training SVM")
    model = train_svm(X_train, y_train)

    print("Step 6: Evaluating")
    evaluate(model, X_test, y_test)

    print("Completed Successfully")


if __name__ == "__main__":
    main()