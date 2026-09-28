import logging

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_answer_generator, get_embeddings_service, get_vector_store
from app.infrastructure.openai_answer_generator import OpenAIAnswerGenerator
from app.infrastructure.sentence_transformer_embeddings import LocalSentenceTransformerEmbeddings
from app.infrastructure.qdrant_store import QdrantVectorStore
from app.schemas.rag import QueryRequest, QueryResponse, SearchHit, SearchRequest
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


def get_rag_service(
    embeddings_service: LocalSentenceTransformerEmbeddings = Depends(get_embeddings_service),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    answer_generator: OpenAIAnswerGenerator | None = Depends(get_answer_generator),
) -> RAGService:
    return RAGService(embeddings_service, vector_store, answer_generator)


@router.post("/search", response_model=list[SearchHit])
def search_documents(request: SearchRequest, service: RAGService = Depends(get_rag_service)):
    try:
        return service.search(request.query, request.top_k)
    except Exception as exc:
        logger.exception("RAG search failed")
        raise HTTPException(status_code=503, detail="La búsqueda no está disponible temporalmente.") from exc


@router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, service: RAGService = Depends(get_rag_service)):
    try:
        return service.answer_question(request.query, request.top_k)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=503, detail="La consulta no está disponible temporalmente.") from exc
