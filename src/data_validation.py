import os
import sys
import logging
import pandas as pd

# Ensure Python can find modules from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define expected schema and column rules
REQUIRED_COLUMNS = [
    "Survived", "Pclass", "Name", "Sex", "Age", 
    "SibSp", "Parch", "Ticket", "Fare", "Embarked"
]
VALID_PCLASS = {1, 2, 3}
VALID_SEX = {"male", "female"}


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validates dataset integrity, schema, and column value constraints.
    """
    logging.info("Starting data validation checks...")

    # 1. Check for required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        logging.error(f"Validation failed: Missing columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 2. Check categorical domains
    invalid_pclass = set(df["Pclass"].dropna().unique()) - VALID_PCLASS
    if invalid_pclass:
        logging.error(f"Validation failed: Unexpected Pclass values: {invalid_pclass}")
        raise ValueError(f"Invalid values in Pclass: {invalid_pclass}")

    invalid_sex = set(df["Sex"].dropna().unique()) - VALID_SEX
    if invalid_sex:
        logging.error(f"Validation failed: Unexpected Sex values: {invalid_sex}")
        raise ValueError(f"Invalid values in Sex: {invalid_sex}")

    # 3. Check target column for missing values
    if df["Survived"].isnull().any():
        logging.error("Validation failed: Target column 'Survived' contains null values.")
        raise ValueError("Target column 'Survived' cannot contain null values.")

    # 4. Range checks
    if (df["Fare"] < 0).any():
        logging.error("Validation failed: Found negative values in 'Fare'.")
        raise ValueError("Fare values cannot be negative.")

    logging.info("All data validation checks passed successfully!")
    return True


if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "titanic.csv")
    df = load_data(raw_data_path)
    validate_data(df)
