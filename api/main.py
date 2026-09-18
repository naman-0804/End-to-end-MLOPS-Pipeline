import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Production-ready inference service for Titanic survival prediction.",
    version="1.0.0"
)

# 2. Path to our saved model artifact
MODEL_PATH = os.path.join("models", "model.joblib")

# Load model globally on startup so it's ready in memory
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train it first using src/train.py!")

model = joblib.load(MODEL_PATH)


# 3. Define Request Body schema using Pydantic
class PassengerInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)", example=3)
    Sex: str = Field(..., description="Gender: 'male' or 'female'", example="female")
    Age: float = Field(..., ge=0, le=120, description="Age in years", example=22.0)
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard", example=1)
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard", example=0)
    Fare: float = Field(..., ge=0.0, description="Passenger fare", example=7.25)
    Embarked: str = Field(..., description="Port of Embarkation: 'C', 'Q', or 'S'", example="S")


# 4. Define Response Body schema
class PredictionResponse(BaseModel):
    survived: bool
    survival_probability: float


@app.get("/")
def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy", "service": "titanic-survival-prediction"}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerInput):
    """
    Accepts passenger details and returns survival prediction.
    """
    try:
        # Convert incoming JSON to a 1-row DataFrame
        input_data = pd.DataFrame([{
            "Pclass": passenger.Pclass,
            "Age": passenger.Age,
            "SibSp": passenger.SibSp,
            "Parch": passenger.Parch,
            "Fare": passenger.Fare,
            # Match the exact one-hot encoded dummy columns our model was trained on:
            "Sex_male": 1 if passenger.Sex.lower() == "male" else 0,
            "Embarked_Q": 1 if passenger.Embarked.upper() == "Q" else 0,
            "Embarked_S": 1 if passenger.Embarked.upper() == "S" else 0,
        }])

        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return PredictionResponse(
            survived=bool(prediction == 1),
            survival_probability=round(float(probability), 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
