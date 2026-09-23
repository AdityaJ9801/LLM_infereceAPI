import asyncio
import gc
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
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


class ServerBusyError(Exception):
    """Raised when the request queue is already at capacity."""


class GPUOutOfMemoryError(Exception):
    """Raised when a generation hits a CUDA OOM - e.g. while probing how high
    MAX_CONCURRENT_REQUESTS can go. Recoverable: the allocator is cleared
    before this is raised, so subsequent requests aren't affected."""


def _dtype_from_setting(name: str):
    if name == "auto":
        return "auto"
    return getattr(torch, name)


class LLMEngine:
    """transformers.generate() wrapper with bounded concurrency, a request
    queue, and idle VRAM release.

    - Up to `max_concurrent_requests` generations run on the GPU at once
      (each with its own KV cache); beyond that, requests wait on a
      semaphore. Beyond `max_queue_size` waiting, new requests are rejected
      with ServerBusyError instead of queueing indefinitely.
    - If nothing runs for `idle_unload_seconds`, a background task frees the
      model from VRAM; the next request transparently reloads it.

    Race to guard against: the idle-unloader must never free the model while
    a request is using it, or in the small window where a request has
    decided the model is loaded but hasn't started using it yet. Both the
    "start using the model" step and "unload the model" step take
    `_load_lock` and check/mutate `_active` inside it, so they can't
    interleave - a request either fully wins the race (model is confirmed
    loaded and `_active` is incremented before anything can unload it) or
    fully loses it (waits for the unload to finish, then reloads).
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._active = 0
        self._queue_waiting = 0
        self._last_used = time.monotonic()
        self._idle_task: Optional[asyncio.Task] = None

    async def startup(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.tokenizer_name or settings.model_name,
            trust_remote_code=settings.trust_remote_code,
        )
        await self._load_model()
        if settings.idle_unload_seconds > 0:
            self._idle_task = asyncio.create_task(self._idle_watcher())

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()

    def status(self) -> dict:
        return {
            "model_loaded": self.model is not None,
            "active_requests": self._active,
            "queued_requests": self._queue_waiting,
        }

    def check_admission(self) -> None:
        """Cheap synchronous pre-check, called before committing to a
        StreamingResponse - once a stream starts, the 200 status is already
        sent, so a ServerBusyError raised later inside generate_stream()
        can't be turned into a clean 503 anymore. _slot() re-checks this
        properly for the non-streaming path; this is a best-effort early
        rejection for both paths (small race, acceptable for a soft limit)."""
        if self._queue_waiting >= settings.max_queue_size:
            raise ServerBusyError(
                f"Server busy: {settings.max_queue_size} requests already queued, try again shortly"
            )

    async def _load_model(self) -> None:
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

        model = await asyncio.to_thread(
            AutoModelForCausalLM.from_pretrained, settings.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        self._last_used = time.monotonic()
        logger.info("Model loaded on %s", self.model.device)

    async def _unload_model(self) -> None:
        async with self._load_lock:
            if self._active > 0 or self.model is None:
                return
            idle_for = time.monotonic() - self._last_used
            logger.info("Unloading model from VRAM (idle for %.0fs)", idle_for)
            del self.model
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

    async def _idle_watcher(self) -> None:
        while True:
            await asyncio.sleep(settings.idle_check_interval_seconds)
            if self.model is None or self._active > 0:
                continue
            if time.monotonic() - self._last_used >= settings.idle_unload_seconds:
                await self._unload_model()

    @asynccontextmanager
    async def _slot(self):
        if self._queue_waiting >= settings.max_queue_size:
            raise ServerBusyError(
                f"Server busy: {settings.max_queue_size} requests already queued, try again shortly"
            )
        self._queue_waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._queue_waiting -= 1

        async with self._load_lock:
            if self.model is None:
                await self._load_model()
            self._active += 1
        try:
            yield
        finally:
            self._active -= 1
            self._last_used = time.monotonic()
            self._semaphore.release()

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
        async with self._slot():
            return await asyncio.to_thread(
                self._generate_sync, messages, max_new_tokens, temperature, top_p, stop
            )

    def _generate_sync(self, messages, max_new_tokens, temperature, top_p, stop) -> dict:
        inputs = self._build_inputs(messages)
        prompt_len = inputs["input_ids"].shape[-1]
        gen_kwargs = self._gen_kwargs(inputs, max_new_tokens, temperature, top_p, stop)
        try:
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise GPUOutOfMemoryError(
                "GPU ran out of memory for this request. Lower MAX_CONCURRENT_REQUESTS, "
                "or retry with a shorter prompt/max_tokens."
            )
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
        async with self._slot():
            inputs = self._build_inputs(messages)
            # timeout=... lets the consumer loop notice a dead generation
            # thread (e.g. after a CUDA OOM) instead of blocking forever on a
            # streamer that will never receive its "done" signal.
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10.0
            )
            gen_kwargs = self._gen_kwargs(inputs, max_new_tokens, temperature, top_p, stop)
            gen_kwargs["streamer"] = streamer

            errors: list = []
            thread = threading.Thread(target=self._run_generate, args=(errors,), kwargs=gen_kwargs, daemon=True)
            thread.start()

            loop = asyncio.get_event_loop()
            while True:
                chunk = await loop.run_in_executor(None, _next_chunk, streamer)
                if chunk is _DONE:
                    break
                if chunk is _EMPTY:
                    if not thread.is_alive():
                        break
                    continue
                yield chunk

            await loop.run_in_executor(None, thread.join)
            if errors:
                raise errors[0]

    def _run_generate(self, errors: list, **kwargs) -> None:
        try:
            with torch.no_grad():
                self.model.generate(**kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            errors.append(GPUOutOfMemoryError(
                "GPU ran out of memory for this request. Lower MAX_CONCURRENT_REQUESTS, "
                "or retry with a shorter prompt/max_tokens."
            ))
        except Exception as e:  # noqa: BLE001 - surfaced to the caller via `errors`
            errors.append(e)


_EMPTY = object()


def _next_chunk(streamer: TextIteratorStreamer):
    try:
        return next(streamer)
    except StopIteration:
        return _DONE
    except queue.Empty:
        return _EMPTY


llm_engine = LLMEngine()
