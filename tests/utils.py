from unittest.mock import patch

import numpy as np
import tensorflow as tf
from tensorflow import keras


# Create a dummy model for unit tests so we don't depend upon an existing model
def model():
    # Create dummy data following the shape eexpected by the real model
    X = np.array([-1.0, 0.0], dtype=float)
    y = np.array([-2.0, 1.0], dtype=float)

    # Train dummy model
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(X, y, epochs=1)

    return model


# Path model loading with the dummy one
with patch("src.predict.load_keras_model", return_value=model()):
    from src.app import app  # noqa: F401
