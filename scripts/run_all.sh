#!/bin/bash

set -xe

echo "###############################"
echo "Training Stage"
echo "###############################"
pip install -r requirements_dev.txt
python src/train.py

echo "###############################"
echo "Unitary Testing Stage"
echo "###############################"
pre-commit install
pytest tests

echo "###############################"
echo "Integration Testing Stage"
echo "###############################"
docker build . -t service-mlops:0.0.1
pytest testIT

echo "###############################"
echo "Load Testing Stage"
echo "###############################"
docker run -it -d --name load-test --rm -p 8000:8000 -v $PWD/my_new_best_model.h5:/opt/app/model.h5 -e MODEL_PATH=model.h5 service-mlops:0.0.1
sleep 10
locust --config load_test/load_test.conf
docker stop load-test
