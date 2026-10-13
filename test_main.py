import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict(client):

    response = client.post(
        "/predict",
        json={"text": "I was charged twice"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] in [
        "Billing", "Technical", "Account"
    ]

    assert data["priority"] in [
        "Low", "Medium", "High"
    ]


def test_empty_text(client):

    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_spaces_only(client):

    response = client.post(
        "/predict",
        json={"text": "   "}
    )

    assert response.status_code == 422


def test_missing_text(client):

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


def test_confidence(client):

    response = client.post(
        "/predict",
        json={"text": "My application is crashing"}
    )

    assert response.status_code == 200

    confidence = response.json()["confidence"]

    assert 0 <= confidence <= 1


def test_batch(client):

    response = client.post(
        "/predict-batch",
        json=[
            {"text": "My refund is delayed"},
            {"text": "I cannot log in"}
        ]
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2


def test_empty_batch(client):

    response = client.post(
        "/predict-batch",
        json=[]
    )

    assert response.status_code == 422