#!/bin/bash
echo "Installing project dependencies"
pip install -r requirements_dev.txt

echo "Installing pre-commit"
pre-commit install

echo "Running tests"
pytest tests
