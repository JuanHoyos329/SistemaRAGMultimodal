from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    id: str
    score: float
    document_id: str
    filename: str
    page_number: int
    content: str
    chunk_type: str = "text"
    bbox: list[float] | None = None
    associated_image_path: str | None = None
    image_url: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class QueryRequest(SearchRequest):
    history: list[ChatMessage] = Field(default_factory=list, max_length=6)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SearchHit]
    warning: str | None = None
