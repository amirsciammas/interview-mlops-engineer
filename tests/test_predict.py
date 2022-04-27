from tempfile import TemporaryDirectory

import numpy as np
from tensorflow import keras

from src import predict


def test_load_keras_model():

    """
    This should test that the model is loaded correctly
    """

    inp = keras.Input((64, 64, 1))
    out = keras.layers.Lambda(lambda x: x)(inp)
    model = keras.Model(inp, out)

    # Save a dummy model in a temporary directory and check that has been saved
    with TemporaryDirectory() as d:
        path = f"{d}/my_model.h5"
        model.save(path)
        new_model = predict.load_keras_model(path)

    assert isinstance(new_model, keras.Model)


def test_predict():

    """
    This should test that the model can get input data and make predictions
    """

    class ModelStub:
        def predict(*args, **kwargs):
            return np.array([[1]])

    result = predict.predict([1], ModelStub())

    assert isinstance(result, np.ndarray)
