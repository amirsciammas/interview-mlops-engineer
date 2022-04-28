#!/bin/bash

docker build . -t service-mlops:0.0.1
pytest testIT
