import re
from typing import Any

from qdrant_client import QdrantClient, models
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config.settings import settings
from app.entities.chunk import Chunk


class QdrantVectorStore:
    def __init__(self, host: str = settings.QDRANT_HOST, port: int = settings.QDRANT_PORT):
        self.client = QdrantClient(host=host, port=port, timeout=10)
        self.collection_name = settings.QDRANT_COLLECTION
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=settings.EMBEDDING_DIMENSION, distance=models.Distance.COSINE),
            )
        # Full text index powers the lexical half of hybrid retrieval.
        collection = self.client.get_collection(self.collection_name)
        if "content" not in (collection.payload_schema or {}):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=32,
                    lowercase=True,
                ),
                wait=True,
            )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            points.append(models.PointStruct(
                id=chunk.id,
                vector=chunk.embedding,
                payload={
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "content": chunk.content,
                    "bbox": chunk.bbox,
                    "associated_image_path": chunk.associated_image_path,
                    "chunk_type": chunk.chunk_type,
                    "metadata": chunk.metadata,
                },
            ))
        for offset in range(0, len(points), 100):
            self._upsert_batch(points[offset:offset + 100])

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=8), reraise=True)
    def _upsert_batch(self, points: list[models.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search_chunks(self, query_vector: list[float], query: str, top_k: int = 5) -> list[dict[str, Any]]:
        candidate_limit = min(max(top_k * 4, 20), 100)
        semantic = self._query_points(
            collection_name=self.collection_name, query=query_vector, limit=candidate_limit, with_payload=True,
        ).points
        terms = list(dict.fromkeys(re.findall(r"[\w-]{2,}", query.lower(), flags=re.UNICODE)))[:12]
        lexical = []
        if terms:
            conditions = [models.FieldCondition(key="content", match=models.MatchText(text=term)) for term in terms]
            lexical, _ = self._scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(should=conditions),
                limit=candidate_limit,
                with_payload=True,
            )
            lexical.sort(
                key=lambda point: sum((point.payload or {}).get("content", "").lower().count(term) for term in terms),
                reverse=True,
            )

        # Reciprocal-rank fusion gives semantic and exact-term retrieval equal opportunity.
        fused: dict[str, tuple[float, Any]] = {}
        semantic_scores: dict[str, float] = {}
        for ranked in (semantic, lexical):
            for rank, point in enumerate(ranked, start=1):
                key = str(point.id)
                score, _ = fused.get(key, (0.0, point))
                fused[key] = (score + 1 / (60 + rank), point)
                if ranked is semantic:
                    semantic_scores[key] = float(point.score or 0.0)
        ordered = sorted(fused.values(), key=lambda item: item[0], reverse=True)[:top_k]
        hits = []
        for score, point in ordered:
            payload = point.payload or {}
            hits.append({
                "id": str(point.id), "score": float(score),
                "document_id": payload.get("document_id", ""),
                "filename": payload.get("filename", "documento.pdf"),
                "page_number": payload.get("page_number", 0),
                "content": payload.get("content", ""),
                "chunk_type": payload.get("chunk_type", "text"),
                "bbox": payload.get("bbox"),
                "associated_image_path": payload.get("associated_image_path"),
                "semantic_score": semantic_scores.get(str(point.id), 0.0),
            })
        return hits

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=8), reraise=True)
    def _query_points(self, **kwargs):
        return self.client.query_points(**kwargs)

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=8), reraise=True)
    def _scroll(self, **kwargs):
        return self.client.scroll(**kwargs)
