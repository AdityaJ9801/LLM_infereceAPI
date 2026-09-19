import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

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


def _stop_list(stop) -> Optional[List[str]]:
    if stop is None:
        return None
    return [stop] if isinstance(stop, str) else list(stop)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.model_name}


@app.get("/v1/models")
async def list_models(_: None = Depends(check_api_key)):
    return {"object": "list", "data": [{"id": settings.model_name, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _: None = Depends(check_api_key)):
    messages = [m.model_dump() for m in req.messages]
    max_new_tokens = req.max_tokens or settings.default_max_tokens
    stop = _stop_list(req.stop)
    model_name = req.model or settings.model_name

    if req.stream:
        return StreamingResponse(
            _stream_generator(messages, max_new_tokens, req.temperature, req.top_p, stop, model_name),
            media_type="text/event-stream",
        )

    result = await llm_engine.generate(messages, max_new_tokens, req.temperature, req.top_p, stop)

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result["text"]),
                finish_reason=result["finish_reason"],
            )
        ],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        ),
    )


async def _stream_generator(messages, max_new_tokens, temperature, top_p, stop, model_name):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

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

    async for delta_text in llm_engine.generate_stream(messages, max_new_tokens, temperature, top_p, stop):
        if not delta_text:
            continue
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

    yield _sse(
        ChatCompletionStreamResponse(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(index=0, delta=DeltaMessage(), finish_reason="stop")
            ],
        )
    )
    yield "data: [DONE]\n\n"


def _sse(payload) -> str:
    return f"data: {payload.model_dump_json()}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
