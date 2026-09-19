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
    # Real Hugging Face repo IDs only. Default fits comfortably in ~45GB VRAM
    # at plain bf16 (no quantization) - see README for the 32B+4bit upgrade path.
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-14B")
    tokenizer_name: Optional[str] = _opt("TOKENIZER_NAME")
    trust_remote_code: bool = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"
    # Qwen3's chat template has a built-in reasoning ("thinking") mode; harmless
    # extra kwarg for chat templates that don't reference it.
    enable_thinking: bool = os.getenv("ENABLE_THINKING", "true").lower() == "true"

    # "bfloat16", "float16", or "auto" (use the checkpoint's declared dtype).
    torch_dtype: str = os.getenv("TORCH_DTYPE", "bfloat16")
    device: str = os.getenv("DEVICE", "cuda:0")
    # 4-bit (bitsandbytes) quantization - opt-in, needed to fit larger models
    # (e.g. 32B) in ~45GB. Leave false for the safest/simplest path.
    load_in_4bit: bool = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

    api_key: Optional[str] = _opt("API_KEY")


settings = Settings()
