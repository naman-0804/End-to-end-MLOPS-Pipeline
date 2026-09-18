import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV data and validate it.
    """

    if not os.path.exists(file_path):
        logging.error(f"File not found at path: {file_path}")
        raise FileNotFoundError(f"File does not exist: {file_path}")

    logging.info(f"Loading data from {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        logging.error("Loaded dataframe is empty")
        raise ValueError("Dataframe is empty")

    logging.info(f"Successfully loaded dataset with shape: {df.shape}")

    return df


if __name__ == "__main__":

    raw_data_path = os.path.join(
        "data",
        "raw",
        "titanic.csv"
    )

    df = load_data(raw_data_path)

    print(df.head())