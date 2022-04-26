"""This script contains all the functionalities related to APIs for 
serving the model."""

from typing import List, Dict

from fastapi import FastAPI

from model import MyModel

app = FastAPI()

# Load the model from the trained checkpoint
model = MyModel()
model.from_pretrained("my_best_model.h5")


@app.post("/")
def predict(batch: Dict[str, List[float]]) -> Dict[str, List[List[float]]]:
    """Prediction API POST Method.

    Args:
        batch (Dict[str, List[float]]): The input batch.

    Returns:
        Dict[str, List[List[float]]]: The predicted batch.
    """

    batch_data = batch["values"]
    predictions = model.make_prediction(batch_data)
    predictions = {"predictions": predictions.tolist()}
    return predictions
