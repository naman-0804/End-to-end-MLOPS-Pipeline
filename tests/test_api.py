from fastapi.testclient import TestClient
from api.main import app

# TestClient lets us simulate HTTP calls without running a live server
client = TestClient(app)


def test_health_check():
    """Test GET / returns healthy status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "titanic-survival-prediction"}


def test_predict_endpoint_valid_input():
    """Test POST /predict with valid passenger JSON."""
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100.0,
        "Embarked": "S"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "survived" in data
    assert "survival_probability" in data
    assert isinstance(data["survived"], bool)
    assert 0.0 <= data["survival_probability"] <= 1.0


def test_predict_endpoint_invalid_input():
    """Test POST /predict fails when invalid data is passed (e.g. negative age)."""
    invalid_payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": -5.0,  # Negative age should be rejected by Pydantic ge=0
        "SibSp": 0,
        "Parch": 0,
        "Fare": 50.0,
        "Embarked": "S"
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
