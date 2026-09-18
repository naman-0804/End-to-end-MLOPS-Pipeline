import os
import sys
import logging
import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Ensure Python can find modules from project root
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data_ingestion import load_data
from src.data_preprocess import preprocess_data, split_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_model(
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42
):
    """
    Train a Random Forest model and log results to MLflow.
    """

    # Load raw data
    raw_data_path = os.path.join(
        "data",
        "raw",
        "titanic.csv"
    )

    df = load_data(raw_data_path)

    # Preprocess data
    cleaned_df = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(
        cleaned_df,
        random_state=random_state
    )

    # Configure MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Titanic-Survival-Prediction")

    with mlflow.start_run():

        logging.info("Training Random Forest Classifier...")

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        logging.info(
            f"Accuracy={acc:.4f}, "
            f"Precision={prec:.4f}, "
            f"Recall={rec:.4f}, "
            f"F1={f1:.4f}"
        )

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Save model
        os.makedirs("models", exist_ok=True)

        model_path = os.path.join(
            "models",
            "model.joblib"
        )

        joblib.dump(model, model_path)

        logging.info(
            f"Model saved successfully at: {model_path}"
        )

        # Log model artifact
        mlflow.log_artifact(
            model_path,
            artifact_path="model"
        )

        logging.info(
            "Experiment successfully logged to MLflow."
        )

    return model, acc


if __name__ == "__main__":
    train_model(
        n_estimators=100,
        max_depth=5
    )