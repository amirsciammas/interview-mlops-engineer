import os
from typing import Dict

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from app import app
from model import MyModel


@pytest.fixture
def tests_outs_path() -> str:
    OUT_TEST_PATH = "tests/outs"
    if not os.path.exists(OUT_TEST_PATH):
        os.mkdir(OUT_TEST_PATH)
    return OUT_TEST_PATH


@pytest.fixture
def model() -> MyModel:
    model = MyModel()
    model.from_pretrained("tests/test_data/my_best_model.h5")
    return model


@pytest.fixture
def app_client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def test_dataset() -> Dict[str, np.array]:
    data = pd.read_csv("tests/test_data/dataset.csv")
    features = data["x"].to_numpy()
    labels = data["y"].to_numpy()
    return {"features": features, "labels": labels}
