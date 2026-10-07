import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "service" in json_data


def test_optimize_validation_fails_for_single_ticker():
    # portfolio optimization requires at least 2 distinct tickers
    bad_payload = {
        "tickers": ["AAPL"],
        "lookback_years": 2
    }
    response = client.post("/api/v1/optimize", json=bad_payload)
    assert response.status_code == 422


def test_optimize_validation_fails_for_duplicate_tickers():
    bad_payload = {
        "tickers": ["AAPL", "AAPL"],
        "lookback_years": 2
    }
    response = client.post("/api/v1/optimize", json=bad_payload)
    assert response.status_code == 422