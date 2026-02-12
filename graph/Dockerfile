FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN chmod 1777 /tmp && \
    rm -rf /etc/apt/sources.list.d/cuda.list && \
    rm -rf /etc/apt/sources.list.d/nvidia-ml.list

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "numpy<2.0" \
    ultralytics \
    datasets \
    httpx \
    pyarrow \
    "pysbd"

COPY . .
