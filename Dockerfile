FROM python:3.9

# Install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy necessary files
COPY app.py .
COPY model.py .
COPY my_best_model.h5 .

# Set the entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]