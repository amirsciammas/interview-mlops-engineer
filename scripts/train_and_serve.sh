#!/bin/bash
echo "Training model"
python src/train.py

echo "Running API locally"
MODEL_PATH=my_new_best_model.h5 uvicorn --reload src.app:app
