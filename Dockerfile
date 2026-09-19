# Runs on the Linux GPU server with the NVIDIA Container Toolkit installed
# (`nvidia-ctk runtime configure --runtime=docker` + restart docker).
# Plain CUDA + transformers stack (no vLLM) - update the tag below to match
# `nvidia-smi`'s reported CUDA version if it's not 12.8.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV MODEL_NAME=Qwen/Qwen3-14B \
    PORT=8000

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
