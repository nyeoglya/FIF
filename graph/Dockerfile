FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "tqdm" \
    "numpy" \
    "requests" \
    "httpx"

COPY . .
