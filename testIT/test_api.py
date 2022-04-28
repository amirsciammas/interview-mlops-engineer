import os
import time

import docker
import pytest
import requests

URL = "http://localhost:8000"


def wait_for_api(wait: int = 2, timeout: int = 20):
    """
    Wait for the API to start up in the container

    Args:
        wait: time between retries
        timeout: max time to wait
    """

    current_time = time.time()
    max_time = current_time + timeout

    # Keep requesting until timeout
    while current_time < max_time:
        try:
            r = requests.get(f"{URL}/status")
        except Exception as err:
            print(err)
            current_time = time.time()
            time.sleep(wait)
        else:
            if r.status_code == 200:
                break


@pytest.fixture(scope="session")
def api_container():

    """
    Create a fixture to use a container of the API for the integration tests

    Yields:
        The running API container.
    """

    client = docker.from_env()
    root_path = os.path.abspath(".")
    # Run the container at port 8000, the image should exist beforehand
    # We could build it before the tests but the tests would take much longer
    container = client.containers.run(
        image="service-mlops:0.0.1",
        auto_remove=True,
        detach=True,
        environment={"MODEL_PATH": "model.h5"},
        ports={"8000/tcp": 8000},
        volumes=[f"{root_path}/my_new_best_model.h5:/opt/app/model.h5"],
    )

    # Call `/status` to check that the API is up
    wait_for_api()
    yield container

    # Remove after the tests
    container.kill()


def test_api_model_loaded(api_container):

    """
    Should check that the model has been loaded correctly # noqa: DAR101
    """

    assert "Trainable params: 2" in api_container.logs().decode()


def test_api_prediction_one_item():

    """
    Should check that the API retunrs a prediction of one item
    """

    response = requests.post(f"{URL}/predict", json={"data": [10]})
    prediction_response = response.json()

    assert response.status_code == 200
    assert "prediction" in prediction_response
    assert isinstance(prediction_response["prediction"], list)
    assert isinstance(prediction_response["prediction"][0], float)


def test_api_prediction_multiple_items():

    """
    Should check that the API retunrs a prediction of multiple items
    """

    response = requests.post(f"{URL}/predict", json={"data": [10, 11, 12]})
    prediction_response = response.json()

    assert response.status_code == 200
    assert "prediction" in prediction_response
    assert isinstance(prediction_response["prediction"], list)
    assert len(prediction_response["prediction"]) == 3


def test_api_malformed_requests():

    """
    Should test that an unexpected body returns a malformed HTTP status code
    """

    response = requests.post(f"{URL}/predict", json={"random": "body"})

    assert response.status_code == 422
