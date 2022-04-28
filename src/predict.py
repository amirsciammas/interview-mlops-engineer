import logging

from tensorflow.keras.models import load_model

logger = logging.getLogger("prediction")
FORMAT = "%(levelname)s-%(asctime)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)


def load_keras_model(path):

    """
    Function to load a Tensorflow model using Keras API and shoow it's summary

    Args:
        path: location of the trained model

    Returns:
        A trained model.
    """

    # Load model and show summary
    logger.info(f"Loading model from: {path}")
    savedModel = load_model(path)
    savedModel.summary()

    return savedModel


def predict(data, model):

    """
    Function to make predictions using a model

    Args:
        data: input data for the model to make predictions
        model: trained Keras model

    Returns:
        The prediction based on the input data.
    """

    # Predict example
    prediction = model.predict(data)
    logging.info(f"Input: {data} Prediction: {prediction}")

    return prediction


if __name__ == "__main__":
    model = load_keras_model("my_new_best_model.h5")
    predict([10], model)
