# data_loader.py

import pandas as pd
from config import DATASET_PATH


def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    return df