"""This script contains all the functionalities related to the model"""

from typing import List, Optional

import numpy as np
import tensorflow as tf


class MyModel:
    """
    A class that contains all the logic pertaining the model"""

    def __init__(self):
        """Constructor"""

        self._model_arch = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=1, input_shape=[1])]
        )
        self._model = None

    def from_pretrained(self, model_path: str):
        """Loads and initializes a model from a pretraining checkpoint.

        Args:
            model_path (str): Path/to the model to use.
        """

        self._model = tf.keras.models.load_model(model_path)

    def model_info(self) -> str:
        """Returns info about the model.

        Returns:
            str: Model summary.
        """

        self._model.summary()

    def train(self, features: np.array, labels: np.array, epochs: Optional[int] = 500):
        """Trains the model with the given features-labels.

        The model is automatically saved in H5 format.

        Args:
            features (np.array): Input features.
            labels (np.array): Labels.
            epochs (int, Optional): The number of epochs to train.
        """

        self._model_arch.compile(optimizer="sgd", loss="mean_squared_error")
        self._model_arch.fit(features, labels, epochs=epochs)
        self._model_arch.save("my_best_model.h5")

    def make_prediction(self, batch: List[float]) -> List[float]:
        """Inference over a batch of data.

        Args:
            batch (List[float]): The input batch data.

        Returns:
            List[float]: The predicted batch data.
        """

        predicted_batch = self._model.predict(batch)
        return predicted_batch
