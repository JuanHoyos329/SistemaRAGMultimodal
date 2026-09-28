import re
import uuid
from typing import Any

from app.entities.chunk import Chunk
from app.infrastructure.text_normalizer import normalize_extracted_text


class DocumentChunker:
    """Layout-aware chunking: retain block boundaries, merge nearby paragraphs, split oversized blocks."""

    def __init__(self, max_chunk_size: int = 1200, overlap: int = 150):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def create_chunks(
        self, document_id: str, filename: str, extracted_pages: list[dict[str, Any]]
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page in extracted_pages:
            images = page["images"]
            pending: list[dict[str, Any]] = []
            pending_len = 0
            for block in sorted(page["text_blocks"], key=lambda item: (item["bbox"][1], item["bbox"][0])):
                text = re.sub(r"\s+", " ", normalize_extracted_text(block["text"])).strip()
                if not text:
                    continue
                # Do not merge blocks separated by a large vertical layout gap.
                if pending and (block["bbox"][1] - pending[-1]["bbox"][3] > 36 or pending_len + len(text) > self.max_chunk_size):
                    self._emit(chunks, document_id, filename, page["page"], pending, images)
                    pending, pending_len = [], 0
                for piece in self._split(text):
                    if pending and pending_len + len(piece) > self.max_chunk_size:
                        self._emit(chunks, document_id, filename, page["page"], pending, images)
                        pending, pending_len = [], 0
                    pending.append({**block, "text": piece})
                    pending_len += len(piece)
            if pending:
                self._emit(chunks, document_id, filename, page["page"], pending, images)
        return chunks

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.max_chunk_size:
            return [text]
        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            if end < len(text):
                boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
                if boundary > start + self.max_chunk_size // 2:
                    end = boundary + 1
            pieces.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(start + 1, end - self.overlap)
        return [piece for piece in pieces if piece]

    @staticmethod
    def _emit(chunks, document_id, filename, page_number, blocks, images):
        content = "\n\n".join(block["text"] for block in blocks)
        box = [
            min(block["bbox"][0] for block in blocks), min(block["bbox"][1] for block in blocks),
            max(block["bbox"][2] for block in blocks), max(block["bbox"][3] for block in blocks),
        ]
        def distance(image):
            ibox = image["bbox"]
            dx = max(box[0] - ibox[2], ibox[0] - box[2], 0)
            dy = max(box[1] - ibox[3], ibox[1] - box[3], 0)
            return dx * dx + dy * dy
        nearest = min(images, key=distance) if images else None
        chunks.append(Chunk(
            id=str(uuid.uuid4()), document_id=document_id, filename=filename,
            page_number=page_number, content=content, bbox=box,
            associated_image_path=nearest["image_path"] if nearest and distance(nearest) <= 180**2 else None,
            chunk_type=("table" if any(block["type"] == "table" for block in blocks)
                        else "image_description" if any(block["type"] == "image" for block in blocks)
                        else "text"),
            metadata={"source": filename, "page": page_number},
        ))
