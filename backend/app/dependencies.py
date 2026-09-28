from functools import lru_cache

from app.config.settings import settings
from app.infrastructure.job_store import SQLiteJobStore
from app.infrastructure.openai_answer_generator import OpenAIAnswerGenerator
from app.infrastructure.sentence_transformer_embeddings import LocalSentenceTransformerEmbeddings, load_embeddings_service
from app.infrastructure.qdrant_store import QdrantVectorStore


@lru_cache(maxsize=1)
def get_embeddings_service() -> LocalSentenceTransformerEmbeddings:
    return load_embeddings_service()


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()


@lru_cache(maxsize=1)
def get_job_store() -> SQLiteJobStore:
    return SQLiteJobStore(settings.STORAGE_DIR / "jobs.sqlite3")


@lru_cache(maxsize=1)
def get_answer_generator() -> OpenAIAnswerGenerator | None:
    if not settings.OPENAI_API_KEY:
        return None
    return OpenAIAnswerGenerator()
