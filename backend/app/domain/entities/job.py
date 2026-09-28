from enum import Enum
from typing import Any

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IngestionJobResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus
    progress: float
    error_message: str | None = None
    result_summary: dict[str, Any] | None = None
    created_at: str | None = None
