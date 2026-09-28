import hashlib
from pathlib import Path
from typing import Any

import fitz

from app.infrastructure.text_normalizer import normalize_extracted_text


class PyMuPDFProcessor:
    """Extract text blocks, table text and page images with page-space coordinates."""

    def process_pdf(self, pdf_path: str, document_id: str, image_root: Path) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        image_dir = image_root / document_id
        image_dir.mkdir(parents=True, exist_ok=True)
        image_files: dict[str, str] = {}
        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document):
                blocks: list[dict[str, Any]] = []
                for block in page.get_text("blocks"):
                    if block[6] == 0 and block[4].strip():
                        blocks.append({"text": normalize_extracted_text(block[4]).strip(), "bbox": list(block[:4]), "type": "text"})

                # Tables become readable row text and keep their page coordinates.
                try:
                    tables = page.find_tables().tables
                except Exception:
                    tables = []
                for table in tables:
                    rows = table.extract()
                    table_text = "\n".join(
                        " | ".join(normalize_extracted_text(str(cell or "")).strip() for cell in row)
                        for row in rows
                    )
                    if table_text.strip():
                        blocks.append({"text": f"Tabla:\n{table_text}", "bbox": list(table.bbox), "type": "table"})

                images: list[dict[str, Any]] = []
                seen: set[tuple[int, tuple[float, ...]]] = set()
                for image in page.get_images(full=True):
                    xref = image[0]
                    extracted = document.extract_image(xref)
                    for occurrence, rect in enumerate(page.get_image_rects(xref, transform=False)):
                        box = tuple(round(float(value), 2) for value in rect)
                        identity = (xref, box)
                        if identity in seen or rect.is_empty:
                            continue
                        seen.add(identity)
                        path = image_dir / f"page_{page_index + 1}_img_{xref}_{occurrence}.png"
                        try:
                            pixmap = fitz.Pixmap(extracted["image"])
                            if pixmap.colorspace is not None and pixmap.colorspace.n not in (1, 3):
                                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
                            image_bytes = pixmap.tobytes("png")
                        except Exception:
                            # Some PDF image codecs cannot be exported directly; render the page region.
                            image_bytes = page.get_pixmap(
                                clip=rect, matrix=fitz.Matrix(2, 2), alpha=False
                            ).tobytes("png")
                        image_hash = hashlib.sha256(image_bytes).hexdigest()
                        saved_path = image_files.get(image_hash)
                        if saved_path is None:
                            path.write_bytes(image_bytes)
                            saved_path = str(path)
                            image_files[image_hash] = saved_path
                        images.append({"image_path": saved_path, "image_hash": image_hash, "bbox": list(box)})

                pages.append({"page": page_index + 1, "text_blocks": blocks, "images": images})
        return pages
