"""A simple script for launching a model training"""

import yaml

import pandas as pd

from model import MyModel

if __name__ == "__main__":
    
    # Read parameters
    params = yaml.safe_load(open("params.yaml"))["training"]
    
    # Read the dataset that we will use for training
    data = pd.read_csv("dataset.csv")
    features = data["x"].to_numpy()
    labels = data["y"].to_numpy()
    
    model = MyModel(num_units=params["units"])
    model.train(
        features=features,
        labels=labels,
        output_model_path="my_best_model.h5",
        optimizer=params["optimizer"],
        epochs=params["epochs"]
    )