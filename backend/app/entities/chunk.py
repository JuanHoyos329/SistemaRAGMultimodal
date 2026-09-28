from typing import Any

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    document_id: str
    filename: str
    page_number: int
    content: str
    bbox: list[float] | None = None
    associated_image_path: str | None = None
    chunk_type: str = "text"
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
