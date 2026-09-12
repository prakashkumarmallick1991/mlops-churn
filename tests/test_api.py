from fastapi.testclient import TestClient

from mlops_churn.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_predict():
    payload = {
        "monthly_charges": 50,
        "tenure": 24,
        "support_calls": 2,
        "login_frequency": 10,
        "contract_length": 12,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert "churn" in result
    assert result["churn"] in [0, 1]
