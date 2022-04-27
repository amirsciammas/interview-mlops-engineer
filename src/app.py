from fastapi import FastAPI

from src.predict import load_keras_model, predict
from src.schemas import InputData, ResponseData

# Load model into the application
model = load_keras_model("my_best_model.h5")

description = """
API to make predictions using a Tensorflow model! 🚀

## Predict

You can perform predictions!

"""

app = FastAPI(
    title="MLOps Engineer Predictions API",
    description=description,
    version="0.0.1",
    license_info={
        "name": "MIT License",
    },
)


# Prediction endpoint
@app.post("/predict", status_code=200, response_model=ResponseData)
def predict_model(data: InputData):
    """
    Performs the prediction with the loaded model usingthe request data

    \f
    Args:
        data (InputData): Body that contains the data to use as input

    Returns:
        The predictions made by the model
    """
    return ResponseData(prediction=predict(data.data, model).item(0))
