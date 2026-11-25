import math

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings


def pgvector_literal(embedding: list[float]) -> str:
    expected_dimensions = settings.rag_embedding_dimensions
    if len(embedding) != expected_dimensions:
        raise ValueError(
            "Invalid embedding dimensions: "
            f"expected {expected_dimensions}, "
            f"received {len(embedding)}."
        )
    if not all(math.isfinite(value) for value in embedding):
        raise ValueError("Embedding contains a non-finite value.")

    return (
        "["
        + ",".join(format(value, ".17g") for value in embedding)
        + "]"
    )


class GeminiEmbeddingService:
    def __init__(self) -> None:
        if settings.gemini_api_key is None:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        api_key = settings.gemini_api_key.get_secret_value().strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is empty.")

        self.model = settings.rag_embedding_model
        self.dimensions = settings.rag_embedding_dimensions
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model,
            api_key=api_key,
            output_dimensionality=self.dimensions,
            vertexai=False,
        )

    def embed_document(
        self,
        *,
        content: str,
        title: str,
    ) -> list[float]:
        title = title.strip()
        vectors = self.embeddings.embed_documents(
            [self._validate_text(content)],
            task_type="RETRIEVAL_DOCUMENT",
            titles=[title] if title else None,
        )
        if not vectors:
            raise RuntimeError("Gemini returned no embedding.")
        return self._normalise(vectors[0])

    def embed_query(self, query: str) -> list[float]:
        vector = self.embeddings.embed_query(
            self._validate_text(query),
            task_type="QUESTION_ANSWERING",
        )
        return self._normalise(vector)

    @staticmethod
    def _validate_text(content: str) -> str:
        content = content.strip()
        if not content:
            raise ValueError("Embedding content cannot be empty.")
        return content

    def _normalise(self, embedding: list[float]) -> list[float]:
        if not embedding:
            raise RuntimeError("Gemini returned an empty embedding.")

        if len(embedding) != self.dimensions:
            raise RuntimeError(
                "Unexpected embedding dimensions: "
                f"expected {self.dimensions}, "
                f"received {len(embedding)}."
            )

        if not all(math.isfinite(value) for value in embedding):
            raise RuntimeError("Gemini returned a non-finite embedding.")

        magnitude = math.hypot(*embedding)
        if magnitude == 0:
            raise RuntimeError(
                "Cannot normalise a zero embedding."
            )

        return [value / magnitude for value in embedding]
