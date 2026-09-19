import os
import sys
import pandas as pd
from prefect import flow, task, get_run_logger

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import load_data
from src.data_validation import validate_data
from src.data_preprocess import preprocess_data, split_data
from src.drift_detection import generate_drift_report

import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@task(name="Ingest Raw Data", retries=2, retry_delay_seconds=3)
def ingest_data_task(file_path: str) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Ingesting raw data from: {file_path}")
    df = load_data(file_path)
    logger.info(f"Ingested shape: {df.shape}")
    return df


@task(name="Validate Dataset Integrity")
def validate_data_task(df: pd.DataFrame) -> bool:
    logger = get_run_logger()
    logger.info("Running schema and constraint validation checks...")
    validate_data(df)
    logger.info("Validation passed!")
    return True


@task(name="Preprocess Data")
def preprocess_data_task(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Preprocessing and encoding features...")
    cleaned_df = preprocess_data(df)
    logger.info(f"Cleaned data shape: {cleaned_df.shape}")
    return cleaned_df


@task(name="Train Model & Log to MLflow")
def train_model_task(
    cleaned_df: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42
) -> dict:
    logger = get_run_logger()
    logger.info(f"Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")

    X_train, X_test, y_train, y_test = split_data(cleaned_df, random_state=random_state)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Titanic-Survival-Prediction")

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))

        logger.info(f"Metrics -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

        # MLflow Tracking
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

    return {"accuracy": acc, "f1_score": f1, "model_path": model_path}


@task(name="Monitor Data Drift")
def monitor_drift_task(reference_df: pd.DataFrame, simulate_drift: bool = True) -> str:
    logger = get_run_logger()
    logger.info("Running drift monitoring...")

    current_df = reference_df.copy()
    if simulate_drift:
        current_df["Age"] = current_df["Age"] + 25.0
        current_df["Fare"] = current_df["Fare"] * 3.5

    report_path = "monitoring/drift_report.html"
    generate_drift_report(reference_df, current_df, output_html_path=report_path)
    logger.info(f"Drift report generated at: {report_path}")
    return report_path


@flow(name="titanic-mlops-pipeline", log_prints=True)
def titanic_pipeline_flow(
    raw_data_path: str = "data/raw/titanic.csv",
    n_estimators: int = 100,
    max_depth: int = 5,
    simulate_drift: bool = True
):
    print("🚀 Starting Titanic MLOps Orchestrated Pipeline via Prefect...")

    # 1. Ingest
    raw_df = ingest_data_task(raw_data_path)

    # 2. Validate
    validate_data_task(raw_df)

    # 3. Preprocess
    cleaned_df = preprocess_data_task(raw_df)

    # 4. Train & Track
    metrics = train_model_task(cleaned_df, n_estimators=n_estimators, max_depth=max_depth)

    # 5. Drift Detection
    report = monitor_drift_task(cleaned_df, simulate_drift=simulate_drift)

    print("\n" + "=" * 55)
    print("✅ PREFECT PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📊 Accuracy:     {metrics['accuracy']:.4f}")
    print(f"🎯 F1-Score:     {metrics['f1_score']:.4f}")
    print(f"📁 Model Saved:  {metrics['model_path']}")
    print(f"📈 Drift Report: {report}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    titanic_pipeline_flow()
