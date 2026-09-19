import asyncio
import logging
import os
import threading
from typing import AsyncGenerator, List, Optional

# Must be set before `import torch`. On unprivileged containers attached to an
# NVIDIA MIG slice, NVML device-level queries return NVML_ERROR_NO_PERMISSION,
# which otherwise crashes torch's CUDA caching allocator with a hard assert
# ("NVML_SUCCESS == r INTERNAL ASSERT FAILED") on model load. This forces
# torch to fall back to the standard CUDA Runtime API instead. Harmless on
# non-MIG GPUs too.
os.environ.setdefault("PYTORCH_NO_CUDA_NVML", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from app.config import settings

logger = logging.getLogger("engine")

# Sentinel to distinguish "streamer finished" from a real (possibly empty) chunk,
# since raising StopIteration across an await boundary inside an async generator
# gets converted into a RuntimeError by Python (PEP 479).
_DONE = object()


def _dtype_from_setting(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


class LLMEngine:
    """Plain transformers.generate() wrapper - single model instance, requests
    are serialized (no continuous batching like vLLM has)."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        logger.info("Loading model %s ...", settings.model_name)

        model_kwargs = dict(
            dtype=_dtype_from_setting(settings.torch_dtype),
            device_map={"": settings.device},
            trust_remote_code=settings.trust_remote_code,
            low_cpu_mem_usage=True,
            attn_implementation=settings.attn_implementation,
        )

        if settings.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs.pop("dtype", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.tokenizer_name or settings.model_name,
            trust_remote_code=settings.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(settings.model_name, **model_kwargs)
        self.model.eval()
        logger.info("Model loaded on %s", self.model.device)

    def _build_inputs(self, messages: List[dict]) -> dict:
        # return_dict=True is explicit on purpose: apply_chat_template's return
        # type (bare tensor vs BatchEncoding) has varied across transformers
        # versions, which previously caused generate() to choke on a
        # BatchEncoding passed where it expected a plain tensor.
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=settings.enable_thinking,
        )
        return {k: v.to(self.model.device) for k, v in encoded.items()}

    def _gen_kwargs(self, inputs: dict, max_new_tokens: int, temperature: float, top_p: float, stop: Optional[List[str]]):
        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            kwargs["temperature"] = temperature
        if stop:
            stop_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in stop]
            stop_ids = [ids[-1] for ids in stop_ids if ids]
            if stop_ids:
                kwargs["eos_token_id"] = [self.tokenizer.eos_token_id, *stop_ids]
        return kwargs

    async def generate(
        self,
        messages: List[dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> dict:
        async with self._lock:
            return await asyncio.to_thread(
                self._generate_sync, messages, max_new_tokens, temperature, top_p, stop
            )

    def _generate_sync(self, messages, max_new_tokens, temperature, top_p, stop) -> dict:
        inputs = self._build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[-1]
        gen_kwargs = self._gen_kwargs(inputs, max_new_tokens, temperature, top_p, stop)
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        new_tokens = output_ids[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        finish_reason = "length" if len(new_tokens) >= max_new_tokens else "stop"
        return {
            "text": text,
            "prompt_tokens": prompt_len,
            "completion_tokens": len(new_tokens),
            "finish_reason": finish_reason,
        }

    async def generate_stream(
        self,
        messages: List[dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        await self._lock.acquire()
        try:
            inputs = self._build_inputs(messages)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = self._gen_kwargs(inputs, max_new_tokens, temperature, top_p, stop)
            gen_kwargs["streamer"] = streamer

            thread = threading.Thread(target=self._run_generate, kwargs=gen_kwargs, daemon=True)
            thread.start()

            loop = asyncio.get_event_loop()
            while True:
                chunk = await loop.run_in_executor(None, _next_or_done, streamer)
                if chunk is _DONE:
                    break
                yield chunk

            await loop.run_in_executor(None, thread.join)
        finally:
            self._lock.release()

    def _run_generate(self, **kwargs) -> None:
        with torch.no_grad():
            self.model.generate(**kwargs)


def _next_or_done(streamer: TextIteratorStreamer):
    try:
        return next(streamer)
    except StopIteration:
        return _DONE


llm_engine = LLMEngine()
