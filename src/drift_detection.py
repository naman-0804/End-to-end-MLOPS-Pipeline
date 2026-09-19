import os
import sys
import logging
import pandas as pd

# Support both new and older Evidently versions
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
except ModuleNotFoundError:
    from evidently.legacy.report import Report
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

# Ensure Python can find modules from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import load_data
from src.data_preprocess import preprocess_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_drift_report(
    reference_data: pd.DataFrame, 
    current_data: pd.DataFrame, 
    output_html_path: str = "monitoring/drift_report.html"
):
    """
    Compares reference (training) data with current (production) data
    and generates an interactive HTML drift dashboard.
    """
    logging.info("Building Evidently Data Drift and Data Quality Report...")

    # Exclude target 'Survived'
    features = [col for col in reference_data.columns if col != "Survived"]

    ref_df = reference_data[features]
    curr_df = current_data[features]

    # Create clean, focused Data Drift report
    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(reference_data=ref_df, current_data=curr_df)

    # Save HTML
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    report.save_html(output_html_path)

    logging.info(f"Drift report successfully generated at: {output_html_path}")
    return output_html_path


if __name__ == "__main__":
    # 1. Load baseline reference data
    raw_path = os.path.join("data", "raw", "titanic.csv")
    df = load_data(raw_path)
    reference_df = preprocess_data(df)

    # 2. Simulate "Current" production data with intentional drift!
    current_df = reference_df.copy()
    current_df["Age"] = current_df["Age"] + 25.0  # Simulated drift in Age
    current_df["Fare"] = current_df["Fare"] * 3.5  # Simulated drift in Fare

    # 3. Generate drift report
    report_file = generate_drift_report(reference_df, current_df)
    print(f"\nReport ready! Open this file in your browser to view: {os.path.abspath(report_file)}")
