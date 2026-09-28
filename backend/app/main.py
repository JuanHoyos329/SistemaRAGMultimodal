import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import documents, health, rag
from app.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)
app.mount("/media", StaticFiles(directory=settings.STORAGE_DIR), name="media")
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(rag.router)
