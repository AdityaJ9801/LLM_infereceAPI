# Runs on the Linux GPU server with the NVIDIA Container Toolkit installed
# (`nvidia-ctk runtime configure --runtime=docker` + restart docker).
# The base image ships a matching CUDA/torch/vllm/transformers/fastapi stack,
# which is the easiest way to get correct Blackwell (B200) kernel support
# without hand-matching driver/CUDA/torch versions yourself.
FROM vllm/vllm-openai:latest

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app ./app

ENV MODEL_NAME=Qwen/Qwen3-32B \
    PORT=8000

EXPOSE 8000

# The base image sets its own ENTRYPOINT for the built-in OpenAI server;
# clear it so we can run our custom FastAPI app instead.
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
