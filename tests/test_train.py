import os
from tempfile import TemporaryDirectory

import numpy as np
from tensorflow import keras

from src import train


def test_get_data():

    """
    This should test the `get_data` function returns an a tuple with arrays
    """

    result = train.get_data()
    X, y = result

    assert isinstance(result, tuple)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)


def test_train_model():

    """
    This should test that the trained model is a keras Model
    """

    X = np.array([1, 2])
    y = np.array([1, 2])

    model = train.train_model(X, y)

    assert isinstance(model, keras.Model)


def test_save_model():

    """
    This should test that a model is correctly saved
    """

    inp = keras.Input((64, 64, 1))
    out = keras.layers.Lambda(lambda x: x)(inp)
    model = keras.Model(inp, out)

    with TemporaryDirectory() as d:
        path = f"{d}/my_model"
        train.save_model(model, path)

        assert os.path.isfile(f"{path}.h5")
