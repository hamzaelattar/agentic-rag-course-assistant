from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    rewritten_question: str = ""
    selected_tools: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    diagram_code: str | None = None
    answer_valid: bool | None = None
    rewrite_attempts: int = 0


class ConversationCreateResponse(BaseModel):
    thread_id: str


class ConversationSummary(BaseModel):
    thread_id: str
    title: str
    updated_at: str | None = None


class ConversationDetail(ConversationSummary):
    messages: list[dict[str, Any]]
