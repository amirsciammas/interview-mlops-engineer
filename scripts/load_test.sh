#!/bin/bash

docker build . -t service-mlops:0.0.1
docker run -it --rm -p 8000:8000 -v $PWD/my_new_best_model.h5:/opt/app/model.h5 -e MODEL_PATH=model.h5 service-mlops:0.0.1
sleep 10
locust --config load_test/load_test.conf
