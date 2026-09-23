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
    # Real Hugging Face repo IDs only. Default (~16GB bf16 weights) leaves
    # ~29GB free on a 45GB card - Qwen3-14B's ~17GB free was too tight once
    # combined with eager attention's prefill overhead and tool-schema-laden
    # prompts, causing real-world OOMs. See README for the 32B+4bit upgrade path.
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-8B")
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
    # "eager", "sdpa", or "flash_attention_2". Defaults to "eager" because
    # cuDNN's SDPA backend has known bugs/crashes on Blackwell (B200, sm_100)
    # as of this writing - "eager" sidesteps the optimized-kernel backends
    # entirely at the cost of speed. Try "sdpa" once things are stable if you
    # want the performance back and your stack doesn't hit the same bug.
    attn_implementation: str = os.getenv("ATTN_IMPLEMENTATION", "eager")

    # Bounded concurrency: at most this many generate() calls run on the GPU
    # at once (each holds its own KV cache in VRAM). Extra requests wait in a
    # queue up to max_queue_size before getting a 503.
    max_concurrent_requests: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2"))
    max_queue_size: int = int(os.getenv("MAX_QUEUE_SIZE", "20"))

    # If no requests are active for this many seconds, unload the model from
    # VRAM (freeing it for other processes on a shared server); it's
    # transparently reloaded on the next request. 0 disables idle unload.
    idle_unload_seconds: int = int(os.getenv("IDLE_UNLOAD_SECONDS", "600"))
    idle_check_interval_seconds: int = int(os.getenv("IDLE_CHECK_INTERVAL_SECONDS", "60"))

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    default_max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
    default_temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

    api_key: Optional[str] = _opt("API_KEY")


settings = Settings()
