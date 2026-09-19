import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from vllm import SamplingParams

from app.config import settings
from app.engine import llm_engine
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    Usage,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await llm_engine.load()
    yield


app = FastAPI(title="Local LLM Inference API", lifespan=lifespan)


def check_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    if not settings.api_key:
        # No API_KEY configured -> auth disabled (fine for local-only testing,
        # NOT fine once the server is exposed publicly via Cloudflare).
        return
    if authorization != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model_name}


@app.get("/v1/models")
async def list_models(_: None = Depends(check_api_key)):
    return {"object": "list", "data": [{"id": settings.model_name, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _: None = Depends(check_api_key)):
    messages = [m.model_dump() for m in req.messages]
    prompt = llm_engine.build_prompt(messages)

    sampling_params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens or settings.default_max_tokens,
        stop=req.stop,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
    )

    request_id = f"cmpl-{uuid.uuid4().hex}"
    model_name = req.model or settings.model_name

    if req.stream:
        return StreamingResponse(
            _stream_generator(prompt, sampling_params, request_id, model_name),
            media_type="text/event-stream",
        )

    final_output = None
    async for output in llm_engine.generate(prompt, sampling_params, request_id):
        final_output = output

    if final_output is None:
        raise HTTPException(status_code=500, detail="No output generated")

    text = final_output.outputs[0].text
    finish_reason = final_output.outputs[0].finish_reason
    prompt_tokens = len(final_output.prompt_token_ids)
    completion_tokens = len(final_output.outputs[0].token_ids)

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _stream_generator(prompt, sampling_params, request_id, model_name):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    previous_text = ""

    yield _sse(
        ChatCompletionStreamResponse(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
                )
            ],
        )
    )

    async for output in llm_engine.generate(prompt, sampling_params, request_id):
        current_text = output.outputs[0].text
        delta_text = current_text[len(previous_text) :]
        previous_text = current_text
        finish_reason = output.outputs[0].finish_reason

        if delta_text:
            yield _sse(
                ChatCompletionStreamResponse(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0, delta=DeltaMessage(content=delta_text), finish_reason=None
                        )
                    ],
                )
            )

        if finish_reason:
            yield _sse(
                ChatCompletionStreamResponse(
                    id=chunk_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0, delta=DeltaMessage(), finish_reason=finish_reason
                        )
                    ],
                )
            )

    yield "data: [DONE]\n\n"


def _sse(payload) -> str:
    return f"data: {payload.model_dump_json()}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
