from app.services.rag_service import RAGService


class FakeEmbeddings:
    def get_text_embedding(self, text):
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self, hits):
        self.hits = hits

    def search_chunks(self, query_vector, query, top_k):
        assert query_vector == [0.1, 0.2]
        assert query
        return self.hits[:top_k]


class FakeGenerator:
    def __init__(self):
        self.context = None

    def generate(self, question, context):
        self.context = context
        return "El límite es 10 bar [Manual.pdf, página 12]."


def test_answer_uses_retrieved_context_and_returns_cited_visual_source():
    hit = {
        "id": "chunk-1", "score": 0.03, "document_id": "doc-1", "filename": "Manual.pdf",
        "page_number": 12, "content": "Presión máxima: 10 bar.", "chunk_type": "text",
        "bbox": [10, 20, 100, 50], "associated_image_path": "image.png", "image_url": "/media/images/image.png",
    }
    generator = FakeGenerator()
    service = RAGService(FakeEmbeddings(), FakeVectorStore([hit]), generator)

    result = service.answer_question("¿Qué presión máxima tiene?", top_k=3)

    assert "10 bar" in result["answer"]
    assert result["sources"] == [hit]
    assert "página 12" in generator.context
    assert "imagen relacionada" in generator.context


def test_empty_retrieval_does_not_call_generator():
    generator = FakeGenerator()
    service = RAGService(FakeEmbeddings(), FakeVectorStore([]), generator)

    result = service.answer_question("¿Qué dice el manual?")

    assert result["sources"] == []
    assert "información suficiente" in result["answer"]
    assert generator.context is None


def test_no_llm_returns_a_concise_extractive_answer():
    hit = {
        "id": "chunk-2", "score": 0.02, "document_id": "doc-2", "filename": "Guía.pdf",
        "page_number": 4, "content": "Use casco de protección en todas las zonas señalizadas del área industrial.", "chunk_type": "text",
        "bbox": None, "associated_image_path": None, "image_url": None,
    }
    service = RAGService(FakeEmbeddings(), FakeVectorStore([hit]))

    result = service.answer_question("¿Qué casco y protección se requiere?")

    assert "[Guía.pdf, página 4]" in result["answer"]
    assert result["sources"] == [hit]


