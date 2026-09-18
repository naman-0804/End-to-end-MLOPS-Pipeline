import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure Python can find modules from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import load_data
from src.data_validation import validate_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw titanic data using simple pandas operations:
    1. Fills missing values (Age with median, Embarked with mode).
    2. Drops columns that aren't useful (PassengerId, Name, Ticket, Cabin).
    3. Converts Sex and Embarked to numbers using one-hot encoding (get_dummies).
    """
    df = df.copy()

    # 1. Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # 2. Drop columns not needed for modeling
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # 3. Convert text/categorical columns to numbers (One-Hot Encoding)
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True, dtype=int)

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the cleaned dataframe into train and test sets.
    """
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "titanic.csv")
    df = load_data(raw_data_path)
    
    # Run validation
    validate_data(df)

    # Clean the data
    cleaned_df = preprocess_data(df)
    logging.info(f"Cleaned data shape: {cleaned_df.shape}")
    print("\nCleaned Data Preview:")
    print(cleaned_df.head())

    # Split
    X_train, X_test, y_train, y_test = split_data(cleaned_df)
    logging.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
