FROM nvcr.io/nvidia/pytorch:23.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    "transformers==4.41.2" \
    "peft==0.10.0" \
    "accelerate>=0.29.0" \
    "safetensors>=0.4.0" \
    "tokenizers==0.19" \
    "datasets>=2.18.0,<3.0.0" \
    "cairosvg" \
    "fastapi" \
    "uvicorn" \
    "pydantic"

COPY . .
