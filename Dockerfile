FROM python:3.8-slim

WORKDIR /opt/app

COPY src/ src/
COPY requirements_model.txt .

RUN pip install --no-cache -r requirements_model.txt && \
    rm requirements_model.txt

EXPOSE 8000

CMD ["uvicorn", "--host", "0.0.0.0", "src.app:app"]
