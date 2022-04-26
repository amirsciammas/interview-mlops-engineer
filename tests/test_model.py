import os
from typing import Dict

import numpy as np
import pytest

from model import MyModel

def test_predictions(model: MyModel):
    """Makes sure the model generates predictions correctly"""
    
    input = [10.0]
    ground_truth = [[30.99516]]
    
    output = model.make_prediction(input)
    
    assert pytest.approx(output, 0.001) == ground_truth
    
def test_predictions_batch(model: MyModel):
    """Makes sure the model generates predictions from a batch 
    correctly"""
    
    input = [10.0, 5.0]
    ground_truth = [[30.99516], [15.998667]]
    
    output = model.make_prediction(input)
    
    assert pytest.approx(output, 0.001) == ground_truth
    
def test_train_model(test_dataset: Dict[str, np.array], tests_outs_path: str):
    """Makes sure the model training can be launch correctly."""
    
    untrained_model = MyModel()
    
    out_model = os.path.join(tests_outs_path, "test_model.h5")
    
    untrained_model.train(
        features=test_dataset["features"],
        labels=test_dataset["labels"],
        output_model_path=out_model,
        epochs=3)
    
    assert os.path.exists(out_model)