import logging

from tensorflow.keras.models import load_model

logger = logging.getLogger("prediction")
FORMAT = '%(levelname)s-%(asctime)s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

def load_keras_model(path):

    # Load model and show summary
    logger.info(f"Loading model from: {path}")
    savedModel=load_model(path)
    savedModel.summary()

    return savedModel

def predict(data, model):
    # Predict example
    prediction = model.predict(data)
    logging.info(f"Input: {data} Prediction: {prediction}")

    return prediction

if __name__ == "__main__":
    model = load_keras_model("my_new_best_model.h5")
    predict([10], model)