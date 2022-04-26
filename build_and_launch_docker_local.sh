docker build . -t adidas_model_api:latest
echo "Docker image successfully built!"

docker run -d -it -p 9000:8000 --name model_api adidas_model_api:latest
echo "API REST for model inference running in Docker container..."