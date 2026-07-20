import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def perform_eda(df):

    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns)

    print("\nInformation")
    print(df.info())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nClass Distribution")
    print(df["status"].value_counts())

    plt.figure(figsize=(8, 5))

    df["status"].value_counts().plot(kind="bar")

    plt.title("Class Distribution")
    plt.xlabel("Mental Health Status")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig("../outputs/class_distribution.png")
    plt.close()