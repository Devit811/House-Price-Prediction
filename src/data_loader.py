import pandas as pd
from .config import TRAIN_PATH, TEST_PATH

def load_data():
    """
    Load the Kaggle House Prices dataset from local CSV files.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}. "
            "Please place train.csv in the data/ folder."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_PATH}. "
            "Please place test.csv in the data/ folder."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df
