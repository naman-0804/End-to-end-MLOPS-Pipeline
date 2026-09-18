import os
import joblib
import pandas as pd


def test_model_file_exists():
    """Verify the trained model artifact exists on disk."""
    model_path = os.path.join("models", "model.joblib")
    assert os.path.exists(model_path), "Model artifact 'model.joblib' is missing!"


def test_model_prediction_output():
    """Verify model can take a sample input row and output 0 or 1."""
    model_path = os.path.join("models", "model.joblib")
    model = joblib.load(model_path)

    # Dummy sample matching exact model feature columns
    sample = pd.DataFrame([{
        "Pclass": 3,
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Sex_male": 1,
        "Embarked_Q": 0,
        "Embarked_S": 1
    }])

    pred = model.predict(sample)
    assert pred[0] in [0, 1], f"Unexpected prediction value: {pred[0]}"
