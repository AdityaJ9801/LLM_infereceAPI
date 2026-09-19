import logging
from typing import AsyncGenerator, List, Optional

from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from app.config import settings

logger = logging.getLogger("engine")


class LLMEngine:
    """Thin wrapper around vLLM's AsyncLLMEngine plus a chat-template tokenizer."""

    def __init__(self) -> None:
        self.engine: Optional[AsyncLLMEngine] = None
        self.tokenizer = None

    async def load(self) -> None:
        logger.info("Loading model %s ...", settings.model_name)
        engine_args = AsyncEngineArgs(
            model=settings.model_name,
            tokenizer=settings.tokenizer_name or settings.model_name,
            tensor_parallel_size=settings.tensor_parallel_size,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            max_model_len=settings.max_model_len,
            quantization=settings.quantization,
            dtype=settings.dtype,
            trust_remote_code=settings.trust_remote_code,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.tokenizer_name or settings.model_name,
            trust_remote_code=settings.trust_remote_code,
        )
        logger.info("Model loaded.")

    def build_prompt(self, messages: List[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=settings.enable_thinking,
        )

    async def generate(
        self, prompt: str, sampling_params: SamplingParams, request_id: str
    ) -> AsyncGenerator:
        assert self.engine is not None, "Engine not loaded yet"
        async for output in self.engine.generate(prompt, sampling_params, request_id):
            yield output


llm_engine = LLMEngine()
