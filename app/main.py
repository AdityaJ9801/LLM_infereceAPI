import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.engine import GPUOutOfMemoryError, ServerBusyError, llm_engine
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
    await llm_engine.startup()
    yield
    await llm_engine.shutdown()


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
    return {"status": "ok", "model": settings.model_name, **llm_engine.status()}


@app.get("/v1/models")
async def list_models(_: None = Depends(check_api_key)):
    return {"object": "list", "data": [{"id": settings.model_name, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, _: None = Depends(check_api_key)):
    # exclude_none: tool_calls/tool_call_id are absent on most messages, and
    # the chat template shouldn't see explicit nulls for fields it doesn't
    # expect on a given role.
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    max_new_tokens = req.max_tokens or settings.default_max_tokens
    stop = _stop_list(req.stop)
    model_name = req.model or settings.model_name
    tools = [t.model_dump() for t in req.tools] if req.tools else None

    try:
        llm_engine.check_admission()
    except ServerBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if req.stream:
        return StreamingResponse(
            _stream_generator(messages, max_new_tokens, req.temperature, req.top_p, stop, model_name, tools),
            media_type="text/event-stream",
            headers={
                # Without these, intermediate proxies (and some browser/HTTP
                # clients) buffer the whole response before delivering any of
                # it, defeating the point of streaming.
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    try:
        result = await llm_engine.generate(messages, max_new_tokens, req.temperature, req.top_p, stop, tools)
    except ServerBusyError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except GPUOutOfMemoryError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=result["text"] or None,
                    tool_calls=result["tool_calls"],
                ),
                finish_reason=result["finish_reason"],
            )
        ],
        usage=Usage(
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["prompt_tokens"] + result["completion_tokens"],
        ),
    )


async def _stream_generator(messages, max_new_tokens, temperature, top_p, stop, model_name, tools=None):
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

    finish_reason = "stop"
    try:
        async for chunk in llm_engine.generate_stream(messages, max_new_tokens, temperature, top_p, stop, tools):
            if chunk.get("tool_calls"):
                finish_reason = "tool_calls"
                yield _sse(
                    ChatCompletionStreamResponse(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0, delta=DeltaMessage(tool_calls=chunk["tool_calls"]), finish_reason=None
                            )
                        ],
                    )
                )
            elif chunk.get("content"):
                yield _sse(
                    ChatCompletionStreamResponse(
                        id=chunk_id,
                        created=created,
                        model=model_name,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0, delta=DeltaMessage(content=chunk["content"]), finish_reason=None
                            )
                        ],
                    )
                )
    except (GPUOutOfMemoryError, ServerBusyError) as e:
        # The 200 status + some chunks may already be on the wire by now, so
        # this can't become an HTTP error response - surface it as the last
        # SSE frame instead of just dropping the connection.
        logger.warning("Streaming generation failed: %s", e)
        finish_reason = "error"

    yield _sse(
        ChatCompletionStreamResponse(
            id=chunk_id,
            created=created,
            model=model_name,
            choices=[
                ChatCompletionStreamChoice(index=0, delta=DeltaMessage(), finish_reason=finish_reason)
            ],
        )
    )
    yield "data: [DONE]\n\n"


def _sse(payload) -> str:
    return f"data: {payload.model_dump_json()}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
