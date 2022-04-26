"""A simple script for launching a model training"""

import pandas as pd

from model import MyModel

if __name__ == "__main__":
    
    model = MyModel()
    
    data = pd.read_csv("dataset.csv")
    features = data["x"].to_numpy()
    labels = data["y"].to_numpy()
    
    model.train(
        features=features,
        labels=labels,
        output_model_path="my_best_model.h5",
        epochs=500
    )