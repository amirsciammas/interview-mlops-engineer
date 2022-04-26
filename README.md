# Proposal

This repository contains a pipeline with DAG format that automates the following stages:
- Pull new data from a S3 bucket.
- Train the model with that data.
- Persists the trained model on a S3 bucket, so that each specific version of it can be downloaded where needed.
- Containarize an app with an API REST to make predictions with the trained model.
- Serves the model (locally).

The stages are defined programatically on the file `dvc.yaml`.

To reproduce the pipeline you need to do: `dvc repro`

This pipeline caches all the stages, meaning that if we re-run the pipeline only the stages that have changed are executed, speeding up the whole process. These changes can be, for example:
- New data is stored on the S3 bucket, or the existing one has been transformed.
- The hyperparameters for training changes, for example the number of epochs. This sets the baseline for hyperparameter tuning as well. The file ´params.yml´ contains all the hyperpameters that are tracked with the pipeline.
- The code for training changes, for example trying a new type of model or neural network architecture.
- Similarly, if the code of the app for the API REST changes, the Docker image is re-built.

# Consuming the API

The documentation (Swagger) of the API REST can be seen on:
- If running from a Docker container (with ports mapped -p 8000:9000): `http://127.0.0.1:9000/docs`
- If running without Docker: `http://127.0.0.1:8000/docs`

Make sure to have the server running in both cases. The pipeline already runs the server, but in case you want to test it individually:
- From Docker: `sh ./build_and_launch_docker_local.sh`
- Without Docker: `uvicorn app:app --host "0.0.0.0" --port 8000`

To make a prediction with the model, any standard procedural can be used, for example with Python `requests`:

```
import requests

input_value = {"values": [10.0]}

endpoint = "http://127.0.0.1:9000/"
response = requests.post(
    endpoint,
    json=input_value
)

response.json()  # it must output {"predictions": [[30.99516]]}
```

# Code Structure

The code stucture is as follows:
- tests: Directory containing the unit tests.

    - test_data: Directory containing the test data for the unit tests.

    - test_app.py: Unit tests for the app and the API REST.

    - test_model.py: Unit tests for the model training / inference.

- conftest.py: Unit tests fixtures.
- pytest.ini: Config file for testing.
- tracked_models: Trained, tracked and persisted models.
- app.py: Code pertaining the API REST.
- build_and_launch_docker_local.sh: Code for building and running the API REST in a Docker container.
- Dockerfile: The Dockerfile to build a Docker image.
- dvc.yaml: Pipeline with the stages defined.
- model.py: A script containing a class to represent a ML model.
- params.yaml: Parameters for each pipeline stage.
- create_dataset.ipynb: A simple notebook to create a CSV dataset from the samples provided in `My Best Model.ipynb`.
- push_tracked_dataset.sh: The code used for tracking and versioning the dataset in S3.
- requirements.txt: Python requirements.
- track_model.sh: The code used for tracking and persisting a trained model in S3. 
- train_model.py: Script for training the model.

Each file/script is documented by itself, so any details can be understood reading them.

# Tests
To execute the tests, you can do: `pytest`

There are two tests that require a Docker container running the server, by default they are "xfailed" (meaning that we expect them to fail if no Docker is running).

# TODO's

I have working on creating the API REST and the pipeline for this exercise, so there are some things that might be improved: 
- The model is being served locally, so the next step would be to deploy and run the container in a Cloud Provider like AWS or Heroku.
- We would need to take into account the number of users and provide functionalities like Auto Scaling or using more workers in the web server.
- There a few variables harcoded on the code, they should not be like that. 
- A proper logger should be added.
- Assertion and exception handling should be incorporated on the code. 