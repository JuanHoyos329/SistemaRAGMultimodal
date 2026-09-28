import logging
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.config.settings import settings
from app.interfaces.embeddings import EmbeddingsService
from app.interfaces.llm import AnswerGenerator
from app.interfaces.vector_store import VectorStore
from app.infrastructure.text_normalizer import normalize_extracted_text

logger = logging.getLogger(__name__)
_STOP_WORDS = {
    "que", "son", "es", "los", "las", "un", "una", "de", "del", "en", "por", "para",
    "y", "o", "a", "el", "la", "se", "lo", "como", "cual", "sobre", "con", "me", "mi",
    "cuanto", "cuanta", "cuando", "donde", "quien", "puede", "podria", "ser", "hay", "dice",
    "este", "esta", "esto", "estos", "estas",
}


class RAGService:
    def __init__(self, embeddings_service: EmbeddingsService, vector_store: VectorStore, answer_generator: AnswerGenerator | None = None):
        self.embeddings_service = embeddings_service
        self.vector_store = vector_store
        self.answer_generator = answer_generator

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        overview_query = self._is_overview_query(query)
        retrieval_query = "temas principales, objetivo, introducción y contenido general del documento" if overview_query else query
        query_vector = self.embeddings_service.get_text_embedding(retrieval_query)
        results = self.vector_store.search_chunks(
            query_vector=query_vector, query=retrieval_query, top_k=min(top_k * 8, 80)
        )
        # Broad overview questions have no topic words to anchor retrieval. If the
        # first broad search misses, try common document-summary signals before
        # concluding that the collection has no usable content.
        if overview_query and not results:
            retrieval_query = "introducción objetivo propósito temas conceptos principales resumen"
            query_vector = self.embeddings_service.get_text_embedding(retrieval_query)
            results = self.vector_store.search_chunks(
                query_vector=query_vector, query=retrieval_query, top_k=min(top_k * 8, 80)
            )
        unique_results: list[dict[str, Any]] = []
        canonical_contents: list[str] = []
        query_terms = self._content_tokens(query) - _STOP_WORDS
        for result in results:
            result["content"] = self._clean_content(result.get("content", ""))
            content_terms = self._content_tokens(result["content"])
            matched_terms = query_terms & content_terms
            lexical_coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0
            semantic_score = float(result.get("semantic_score", 0.0))
            lexical_match = (
                lexical_coverage >= 0.5
                and (len(query_terms) == 1 or len(matched_terms) >= 2)
            )
            semantic_match = (
                semantic_score >= settings.MIN_SEMANTIC_SCORE
                and bool(matched_terms)
                and lexical_coverage >= 0.25
            )
            if not overview_query and not (lexical_match or semantic_match):
                continue
            canonical = self._canonical_content(result["content"])
            if len(canonical) > 80 and any(
                SequenceMatcher(None, canonical, previous).ratio() >= 0.95
                for previous in canonical_contents
            ):
                continue
            canonical_contents.append(canonical)
            image_path = result.get("associated_image_path")
            if overview_query:
                result["associated_image_path"] = None
            if image_path and not overview_query:
                try:
                    normalized_path = image_path.replace("\\", "/")
                    marker = "/images/"
                    if marker in normalized_path:
                        result["image_url"] = "/media/images/" + normalized_path.split(marker, 1)[1]
                    else:
                        result["image_url"] = "/media/" + Path(image_path).resolve().relative_to(settings.STORAGE_DIR.resolve()).as_posix()
                except ValueError:
                    result.setdefault("image_url", None)
            unique_results.append(result)
            if len(unique_results) >= top_k:
                break
        return unique_results

    def answer_question(
        self, question: str, top_k: int = 5, history: list[dict[str, str]] | None = None
    ) -> dict[str, Any]:
        history = (history or [])[-6:]
        retrieval_query = self._contextualize_followup(question, history)
        sources = self.search(query=retrieval_query, top_k=top_k)
        if not sources:
            return {
                "answer": "No encontré información suficiente en los PDF para responder. Solo respondo preguntas respaldadas por su contenido.",
                "sources": [],
                "warning": None,
            }

        context = "\n\n".join(
            f"[Fuente: {source['filename']}, página {source['page_number']}, tipo {source['chunk_type']}]\n"
            f"{source['content'][:1000]}"
            + ("\n[Hay una imagen relacionada en esta página.]" if source.get("image_url") else "")
            for source in sources[:4]
        )
        answer, warning = self._generate(question, context, sources, history)
        answer = self._ensure_citations(answer, sources)
        return {"answer": answer, "sources": sources, "warning": warning}

    def _generate(
        self, question: str, context: str, sources: list[dict[str, Any]], history: list[dict[str, str]]
    ) -> tuple[str, str | None]:
        if self.answer_generator is None:
            warning = "OpenAI no está configurado; se generó una respuesta extractiva a partir de las fuentes."
            return self._extractive_answer(question, sources), warning
        try:
            generation_question = question
            if history:
                recent = "\n".join(
                    f"{item['role']}: {item['content'][:600]}" for item in history
                    if item.get("role") in {"user", "assistant"} and item.get("content")
                )
                if recent:
                    generation_question = (
                        f"Historial reciente (úsalo solo para resolver referencias; no es evidencia):\n{recent}\n\n"
                        f"Pregunta actual: {question}"
                    )
            return self.answer_generator.generate(generation_question, context), None
        except Exception as exc:
            logger.exception("LLM generation failed")
            error_type = type(exc).__name__
            code = str(getattr(exc, "code", "") or getattr(exc, "type", "")).lower()
            if error_type == "AuthenticationError":
                warning = "OpenAI rechazó la clave API. Verifica que esté vigente y tenga acceso al proyecto."
            elif error_type == "RateLimitError" and ("quota" in code or "billing" in code):
                warning = "OpenAI no tiene cuota disponible para esta solicitud. Revisa el uso y la facturación del proyecto."
            elif error_type == "RateLimitError":
                warning = "OpenAI limitó temporalmente las solicitudes. Espera un momento y vuelve a intentar."
            elif error_type in {"APITimeoutError", "APIConnectionError"}:
                warning = "No se pudo conectar con OpenAI dentro del tiempo esperado. Revisa la conexión e inténtalo de nuevo."
            elif error_type == "NotFoundError":
                warning = "El modelo de OpenAI configurado no está disponible para esta clave. Revisa OPENAI_CHAT_MODEL."
            else:
                warning = f"Falló la generación con OpenAI ({error_type}). Consulta los logs de la API para ver el detalle."
            fallback = self._extractive_answer(question, sources)
            return fallback, warning

    @staticmethod
    def _ensure_citations(answer: str, sources: list[dict[str, Any]]) -> str:
        """Keep only citations that map to retrieved sources and add a fallback citation."""
        citation_pattern = re.compile(
            r"\[([^\],]+),\s*(?:p[aá]gina|p\.?)\s*(\d+)\]", re.IGNORECASE
        )
        valid_sources = {
            (source["filename"].casefold(), int(source["page_number"]))
            for source in sources
        }
        found_valid = False

        def validate(match: re.Match[str]) -> str:
            nonlocal found_valid
            key = (match.group(1).strip().casefold(), int(match.group(2)))
            if key in valid_sources:
                found_valid = True
                return match.group(0)
            return ""

        cleaned = citation_pattern.sub(validate, answer).strip()
        if not found_valid and sources:
            references = "; ".join(
                f"[{source['filename']}, página {source['page_number']}]"
                for source in sources[:2]
            )
            cleaned = f"{cleaned}\n\nFuentes consultadas: {references}".strip()
        return re.sub(r"\s+([,.;])", r"\1", cleaned)

    @classmethod
    def _contextualize_followup(cls, question: str, history: list[dict[str, str]]) -> str:
        previous_question = next(
            (item["content"] for item in reversed(history) if item.get("role") == "user"), None
        )
        if not previous_question:
            return question
        terms = cls._content_tokens(question) - _STOP_WORDS
        normalized = cls._canonical_content(question)
        reference_words = {"eso", "esa", "ese", "esto", "estos", "estas", "ello", "ahi", "alli", "anterior"}
        if (
            len(terms) <= 1
            or reference_words.intersection(terms)
            or re.search(r"^(?:y\b|.*\b(?:su|sus)\b)", normalized)
        ):
            return f"{previous_question}. Seguimiento: {question}"
        return question

    def _extractive_answer(self, question: str, sources: list[dict[str, Any]]) -> str:
        if self._is_overview_query(question):
            summary_terms = {"objetivo", "analisis", "estudia", "describe", "consiste", "consisten", "evalua", "presenta"}
            candidates = []
            for source_rank, source in enumerate(sources):
                sentences = re.split(r"(?<=[.!?])\s+|\n+", source["content"])
                for sentence in sentences:
                    sentence = sentence.strip(" -|;\t")
                    if len(sentence) < 45 or "¿" in sentence or "?" in sentence:
                        continue
                    terms = self._content_tokens(sentence)
                    score = len(terms & summary_terms)
                    candidates.append((score, -source_rank, source, sentence))
            summaries = []
            seen_sentences = set()
            for _, _, source, sentence in sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True):
                canonical = self._canonical_content(sentence)
                if canonical in seen_sentences:
                    continue
                seen_sentences.add(canonical)
                if len(sentence) > 260:
                    sentence = sentence[:257].rsplit(" ", 1)[0] + "..."
                summaries.append(f"{sentence} [{source['filename']}, p. {source['page_number']}]")
                if len(summaries) == 2:
                    break
            if summaries:
                return "El PDF aborda principalmente estos temas: " + " ".join(summaries)
            return "El PDF contiene información relacionada, pero no recuperé texto suficiente para resumirlo con confianza."
        query_words = {
            word for word in re.findall(r"[\wáéíóúüñ]{3,}", question.lower())
            if word not in _STOP_WORDS
        }
        candidates = []
        for source in sources:
            sentences = re.split(r"(?<=[.!?])\s+|\n+", source["content"])
            for sentence in sentences:
                sentence = sentence.strip(" -|;\t")
                if len(sentence) < 30 or sentence.endswith("?"):
                    continue
                words = set(re.findall(r"[\wáéíóúüñ]{3,}", sentence.lower()))
                overlap = len(query_words & words)
                definition_bonus = 2 if re.search(r"\b(consisten en|se define como|se refiere a|son los lugares)\b", sentence, re.I) else 0
                candidates.append((overlap + definition_bonus, source, sentence))
        if not candidates:
            return "Encontré fuentes relacionadas, pero no una frase suficientemente clara para responder. Revisa las páginas citadas."
        _, source, sentence = max(candidates, key=lambda item: (item[0], min(len(item[2]), 320)))
        if len(sentence) > 420:
            sentence = sentence[:417].rsplit(" ", 1)[0] + "..."
        return f"{sentence} [{source['filename']}, página {source['page_number']}]"

    @staticmethod
    def _clean_content(content: str) -> str:
        text = normalize_extracted_text(content)
        text = re.sub(r"(?i)\btabla:\s*", "", text)
        text = re.sub(r"\s*\|\s*(?:\|\s*)+", " ", text)
        text = re.sub(r"\s*\|\s*", " — ", text)
        text = re.sub(r"\s*—\s*Johann A\. Ospina\b.*$", "", text, flags=re.I)
        text = re.sub(r"\s*—\s*\d+\s*/\s*\d+\s*$", "", text)
        return re.sub(r"\s+", " ", text).strip(" -|;\t")

    @staticmethod
    def _canonical_content(content: str) -> str:
        decomposed = unicodedata.normalize("NFKD", content.lower())
        without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
        return " ".join(re.findall(r"[a-z0-9]+", without_accents))

    @classmethod
    def _is_overview_query(cls, query: str) -> bool:
        terms = cls._content_tokens(query) - _STOP_WORDS
        overview_terms = {
            "trata", "tratamiento", "resume", "resumen", "resumir", "contenido", "contiene",
            "tema", "temas", "objetivo", "objetivos", "explica", "aborda", "pdf", "documento",
            "documentos", "manual", "archivo", "habla", "describe", "presenta", "dime", "cuenta",
            "explicame", "general", "principal", "principales", "sobre", "archivo",
        }
        return bool(terms) and terms.issubset(overview_terms)

    @staticmethod
    def _content_tokens(content: str) -> set[str]:
        decomposed = unicodedata.normalize("NFKD", normalize_extracted_text(content).lower())
        without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
        tokens = set(re.findall(r"[a-z0-9]{3,}", without_accents))
        # Basic Spanish plural normalization helps match "patrones" to "patrón" poorly,
        # while preserving common singular terms for exact support checks.
        stems = set()
        for token in tokens:
            stems.add(token[:-2] if token.endswith("es") and len(token) > 5 else token[:-1] if token.endswith("s") and len(token) > 4 else token)
        return tokens | stems
