from typing import Any, Protocol

from app.entities.chunk import Chunk


class VectorStore(Protocol):
    def upsert_chunks(self, chunks: list[Chunk]) -> None: ...

    def search_chunks(self, query_vector: list[float], query: str, top_k: int = 5) -> list[dict[str, Any]]: ...
