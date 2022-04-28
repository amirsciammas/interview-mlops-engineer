#!/bin/bash

TAG=$1

echo "Building image"

docker build . -t service-mlops:$TAG
