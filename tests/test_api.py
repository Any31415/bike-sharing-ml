from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    payload = {
        "season": 1, "holiday": 0, "workingday": 1, "weather": 1,
        "temp": 20.5, "humidity": 60, "windspeed": 10.0,
        "hour": 8, "dayofweek": 2, "month": 6
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_count" in response.json()
    assert response.json()["predicted_count"] >= 0