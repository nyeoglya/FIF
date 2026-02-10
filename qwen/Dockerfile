FROM vllm/vllm-openai:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "qwen-vl-utils>=0.0.8" \
    "pillow" \
    "dynamic_dispatch"

ENV VLLM_USE_FLASHINFER=1

WORKDIR /app
COPY . .
