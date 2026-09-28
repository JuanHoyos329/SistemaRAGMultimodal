import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from app.config.settings import settings
from app.dependencies import get_embeddings_service, get_job_store, get_vector_store
from app.domain.entities.job import IngestionJobResponse, JobStatus, JobStatusResponse
from app.infrastructure.chunker import DocumentChunker
from app.infrastructure.job_store import SQLiteJobStore
from app.infrastructure.openai_image_captioner import OpenAIImageCaptioner
from app.infrastructure.pymupdf_processor import PyMuPDFProcessor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])
processor = PyMuPDFProcessor()
chunker = DocumentChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)


def process_pdf_in_background(
    job_id: str,
    file_path: str,
    filename: str,
    jobs: SQLiteJobStore,
) -> None:
    try:
        # Load model and vector DB inside the worker, never on the upload request path.
        embeddings = get_embeddings_service()
        vector_store = get_vector_store()
        jobs.update(job_id, status=JobStatus.PROCESSING.value, progress=5)
        pages = processor.process_pdf(file_path, job_id, settings.STORAGE_DIR / "images")
        if settings.OPENAI_API_KEY and settings.ENABLE_IMAGE_CAPTIONS:
            captioner = OpenAIImageCaptioner()
            remaining = settings.MAX_IMAGES_PER_DOCUMENT
            captioned_images: set[str] = set()
            for page in pages:
                page_text = " ".join(block["text"] for block in page["text_blocks"])
                for image in page["images"][:remaining]:
                    image_hash = image.get("image_hash", image["image_path"])
                    if image_hash in captioned_images:
                        continue
                    captioned_images.add(image_hash)
                    try:
                        caption = captioner.describe(image["image_path"], page_text)
                        if caption:
                            page["text_blocks"].append({
                                "text": f"Descripción visual del diagrama: {caption}",
                                "bbox": image["bbox"],
                                "type": "image",
                            })
                    except Exception as exc:
                        logger.exception("Image caption failed job_id=%s page=%s", job_id, page["page"])
                        if type(exc).__name__ in {"RateLimitError", "AuthenticationError", "PermissionDeniedError"}:
                            logger.warning("Skipping remaining image captions after provider rejection")
                            remaining = 0
                    remaining -= 1
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break
        jobs.update(job_id, progress=35)
        chunks = chunker.create_chunks(job_id, filename, pages)
        if not chunks:
            raise ValueError("El PDF no contiene texto indexable. Los PDF escaneados requieren OCR.")
        jobs.update(job_id, progress=50)
        vectors = embeddings.get_text_embeddings([chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        vector_store.upsert_chunks(chunks)
        jobs.update(
            job_id,
            status=JobStatus.COMPLETED.value,
            progress=100,
            result_summary={"pages": len(pages), "chunks": len(chunks), "images": sum(len(p["images"]) for p in pages)},
        )
        logger.info("Ingestion completed job_id=%s pages=%s chunks=%s", job_id, len(pages), len(chunks))
    except Exception as exc:
        logger.exception("Ingestion failed job_id=%s", job_id)
        jobs.update(job_id, status=JobStatus.FAILED.value, error_message=str(exc))
    finally:
        Path(file_path).unlink(missing_ok=True)


@router.post("/upload", response_model=IngestionJobResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    jobs: SQLiteJobStore = Depends(get_job_store),
):
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF.")
    job_id = str(uuid.uuid4())
    upload_dir = settings.STORAGE_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{job_id}.pdf"
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    try:
        with file_path.open("wb") as destination:
            while data := await file.read(1024 * 1024):
                total += len(data)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"El PDF supera el límite de {settings.MAX_UPLOAD_MB} MB.")
                destination.write(data)
    except Exception:
        file_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    if total == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    jobs.create({"job_id": job_id, "filename": filename, "status": JobStatus.PENDING.value, "progress": 0})
    background_tasks.add_task(process_pdf_in_background, job_id, str(file_path), filename, jobs)
    return IngestionJobResponse(job_id=job_id, filename=filename, status=JobStatus.PENDING, message="Archivo recibido; ingesta en segundo plano.")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, jobs: SQLiteJobStore = Depends(get_job_store)):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="El ID del trabajo no existe.")
    return JobStatusResponse(**job)
