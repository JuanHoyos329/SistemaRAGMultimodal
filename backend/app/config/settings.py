from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "Sistema RAG Multimodal"
    VERSION: str = "1.0.0"
    STORAGE_DIR: Path = BASE_DIR / "backend" / "storage"
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "multimodal_rag_v2"
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    OPENAI_API_KEY: str = ""
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_VISION_MODEL: str = "gpt-4o-mini"
    ENABLE_IMAGE_CAPTIONS: bool = True
    MAX_IMAGES_PER_DOCUMENT: int = 30
    MIN_SEMANTIC_SCORE: float = 0.62
    MAX_UPLOAD_MB: int = 30
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 150


settings = Settings()
