import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON-encoded, per the OpenAI convention


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    # Set on role="tool" messages, echoing which call this is a result for.
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "default"
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    tools: Optional[List[ToolDef]] = None
    # Accepted for API compatibility but NOT enforced - the model decides
    # whether to call a tool on its own. See README's tool-calling section.
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
