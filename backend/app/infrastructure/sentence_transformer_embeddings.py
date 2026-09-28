from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config.settings import settings
from app.interfaces.embeddings import EmbeddingsService


class LocalSentenceTransformerEmbeddings(EmbeddingsService):
    """Multilingual text encoder; image context is linked through visual page references."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def get_text_embedding(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def get_image_embedding(self, image_path: str) -> list[float]:
        # Qdrant uses one shared text-vector space. Image files are surfaced alongside
        # the nearest layout chunk; they are not falsely represented as text vectors.
        return []


@lru_cache(maxsize=1)
def load_embeddings_service() -> LocalSentenceTransformerEmbeddings:
    return LocalSentenceTransformerEmbeddings()
