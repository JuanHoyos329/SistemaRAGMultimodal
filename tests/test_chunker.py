from app.infrastructure.chunker import DocumentChunker


def test_chunker_preserves_layout_metadata_and_links_nearest_image():
    pages = [{
        "page": 3,
        "text_blocks": [
            {"text": "Figura 2: flujo del sistema", "bbox": [20, 20, 180, 50], "type": "text"},
            {"text": "La bomba entrega presión constante.", "bbox": [20, 55, 180, 85], "type": "text"},
        ],
        "images": [{"image_path": "storage/doc/page3.png", "bbox": [20, 90, 180, 200]}],
    }]

    chunks = DocumentChunker(max_chunk_size=500).create_chunks("doc-1", "manual.pdf", pages)

    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].filename == "manual.pdf"
    assert chunks[0].bbox == [20, 20, 180, 85]
    assert chunks[0].associated_image_path == "storage/doc/page3.png"
    assert "bomba" in chunks[0].content


def test_chunker_splits_large_blocks_with_bounded_piece_size():
    text = "Una oración técnica. " * 100
    pages = [{"page": 1, "text_blocks": [{"text": text, "bbox": [0, 0, 100, 100], "type": "text"}], "images": []}]

    chunks = DocumentChunker(max_chunk_size=120, overlap=20).create_chunks("doc", "long.pdf", pages)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 120 for chunk in chunks)
