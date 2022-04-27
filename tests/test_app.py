from fastapi.testclient import TestClient

from .utils import app

client = TestClient(app)


def test_prediction_ok():

    """
    This should test that a prediction is succesfully returned
    """

    response = client.post("/predict", json={"data": [10]})

    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_malformed_body():

    """
    This should test that a malformed requests returns a 422
    """

    response = client.post("/predict", json={"data": "a string"})

    assert response.status_code == 422
