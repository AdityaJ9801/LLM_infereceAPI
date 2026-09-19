import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _opt(name: str) -> Optional[str]:
    val = os.getenv(name)
    return val if val else None


@dataclass
class Settings:
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-32B")
    tokenizer_name: Optional[str] = _opt("TOKENIZER_NAME")
    quantization: Optional[str] = _opt("QUANTIZATION")  # "awq", "gptq", "fp8", or None
    dtype: str = os.getenv("DTYPE", "auto")
    trust_remote_code: bool = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
    # Qwen3's chat template has a built-in reasoning ("thinking") mode; harmless
    # extra kwarg for chat templates that don't reference it.
    enable_thinking: bool = os.getenv("ENABLE_THINKING", "true").lower() == "true"

    tensor_parallel_size: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    gpu_memory_utilization: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.92"))
    max_model_len: int = int(os.getenv("MAX_MODEL_LEN", "8192"))

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

    api_key: Optional[str] = _opt("API_KEY")


settings = Settings()
