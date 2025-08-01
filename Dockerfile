FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y     libglib2.0-0     libsm6     libxext6     libxrender-dev     libgomp1     libgl1-mesa-glx     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Preload models before starting the server
RUN python preload_models.py

ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 2 --timeout 300 cloud_backend:app