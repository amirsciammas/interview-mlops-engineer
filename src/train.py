import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger("training")
FORMAT = "%(levelname)s-%(asctime)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def get_data():

    logger.info("Getting data...")
    # Provide the data
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
    logger.info("Success!")

    return xs, ys


def train_model(X, y, optimizer="sgd", loss="mean_squared_error"):

    """
    Function to train a Tensorflow model using Keras API

    Args:
        X: training data
        y: labels for the training data
        optimizer: optimizer to use in training
        loss: loss to measure the performance of the model

    Returns:
        A trained model.
    """

    # Define and compile the neural network
    logger.info("Creating neural network")
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer=optimizer, loss=loss)

    logger.info("Training neural network")
    # Train the neural network
    model.fit(X, y, epochs=500)

    return model


def save_model(model, name):

    """
    Function to save a Tensorflow model using Keras API

    Args:
        model: fitted model instance
        name: name used to save the model
    """
    # Save model into file
    model.save(f"{name}.h5")
    logger.info(f"Model saved as: {name}.h5")


if __name__ == "__main__":
    X, y = get_data()
    model = train_model(X, y)
    save_model(model, "my_new_best_model")
