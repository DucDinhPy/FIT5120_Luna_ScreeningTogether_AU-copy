import hashlib
import math
import re
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from itertools import islice
from pathlib import Path

from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, HttpUrl, field_validator
from pypdf.errors import FileNotDecryptedError, PdfReadError
from sqlalchemy import text

from config import settings
from db.session import SessionLocal
from services.chatbot.rag_embedding import GeminiEmbeddingService, pgvector_literal


CHUNK_SIZE = 1000
PIPELINE_VERSION = "semantic-pdf-v1"
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
PAGE_FOOTER = re.compile(r"page\s+\d+\s+of\s+\d+", re.I)
PAGE_NUMBER = re.compile(r"(?:page\s*)?\d+", re.I)


class DocumentMetadata(BaseModel):
    title: str = Field(min_length=2, max_length=300)
    publisher: str = Field(min_length=2, max_length=200)
    source_url: HttpUrl
    published_at: date | None = None

    @field_validator("title", "publisher", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


@dataclass(frozen=True)
class PDFIngestResult:
    document_id: int
    page_count: int
    indexed_chunks: int
    filtered_chunks: int


def _is_indexable(content: str) -> bool:
    return len(content) >= 100 and len(content.split()) >= 20


class SmartPDFProcessor:
    """Prepare semantic chunks and optionally index them with the configured RAG model."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        threshold: float = 0.7,
    ) -> None:
        self.text_splitter = ThresholdSemanticChunker(model_name, api_key, threshold)

    def process_pdf(self, pdf_path: str | Path) -> list[Document]:
        """Return indexable PDF chunks for inspection."""
        chunks, _, _ = self._process_pdf(self._read_pdf(Path(pdf_path)))
        return chunks

    def ingest_pdf(self, pdf_path: str | Path, metadata: DocumentMetadata) -> int:
        """Index a local PDF in pgvector and return its document ID."""
        path = Path(pdf_path)
        if len(path.name) > 255:
            raise ValueError("The PDF file name must be at most 255 characters.")
        return self._index_pdf(path.name, self._read_pdf(path), metadata).document_id

    def ingest_uploaded_pdf(
        self,
        *,
        file_name: str,
        pdf_data: bytes,
        metadata: DocumentMetadata,
    ) -> PDFIngestResult:
        """Index uploaded PDF bytes and return the indexing summary."""
        if (
            len(file_name) > 255
            or "/" in file_name
            or "\\" in file_name
            or "\x00" in file_name
            or Path(file_name).suffix.casefold() != ".pdf"
        ):
            raise ValueError("The uploaded PDF file name is invalid.")
        return self._index_pdf(file_name, self._pdf_blob(pdf_data, file_name), metadata)

    def _index_pdf(
        self, file_name: str, blob: Blob, metadata: DocumentMetadata
    ) -> PDFIngestResult:
        chunks, page_count, filtered_chunks = self._process_pdf(blob)
        embedding_service = GeminiEmbeddingService()
        vectors = [
            pgvector_literal(
                embedding_service.embed_document(
                    content=chunk.page_content,
                    title=metadata.title,
                )
            )
            for chunk in chunks
        ]
        document_id = self._save_document(
            file_name=file_name,
            file_hash=hashlib.sha256(blob.as_bytes()).hexdigest(),
            metadata=metadata,
            chunks=chunks,
            vectors=vectors,
        )
        return PDFIngestResult(
            document_id=document_id,
            page_count=page_count,
            indexed_chunks=len(chunks),
            filtered_chunks=filtered_chunks,
        )

    @classmethod
    def _read_pdf(cls, path: Path) -> Blob:
        if path.suffix.lower() != ".pdf":
            raise ValueError("Expected a .pdf file.")
        with path.expanduser().open("rb") as pdf:
            data = pdf.read(settings.rag_upload_max_bytes + 1)
        return cls._pdf_blob(data, str(path))

    @staticmethod
    def _pdf_blob(data: bytes, source: str) -> Blob:
        if len(data) > settings.rag_upload_max_bytes:
            raise ValueError(
                f"PDFs are limited to {settings.rag_upload_max_bytes} bytes."
            )
        if not data.lstrip().startswith(b"%PDF-"):
            raise ValueError("The file is not a valid PDF.")
        return Blob.from_data(data, path=source, mime_type="application/pdf")

    def _process_pdf(self, blob: Blob) -> tuple[list[Document], int, int]:
        max_pages = settings.rag_upload_max_pages
        try:
            with closing(PyPDFParser().lazy_parse(blob)) as parsed:
                pages = list(islice(parsed, max_pages + 1))
        except FileNotDecryptedError as exc:
            raise ValueError("Password-protected PDFs are not supported.") from exc
        except PdfReadError as exc:
            raise ValueError("The PDF could not be read.") from exc

        if len(pages) > max_pages:
            raise ValueError(f"PDFs are limited to {max_pages} pages.")
        if not any(page.page_content.strip() for page in pages):
            raise ValueError("No selectable text was found. The PDF may require OCR.")

        generated = self.text_splitter.split_documents(self._process_pages(pages))
        chunks = self._filter_indexable_chunks(generated)
        if not chunks:
            raise ValueError("No indexable text was found in the PDF.")
        if len(chunks) > settings.rag_upload_max_chunks:
            raise ValueError(
                f"PDFs are limited to {settings.rag_upload_max_chunks} chunks."
            )
        return chunks, len(pages), len(generated) - len(chunks)

    def _process_pages(self, pages: list[Document]) -> list[Document]:
        cleaned_pages = [
            Document(
                page_content=self._clean_text(page.page_content),
                metadata={
                    **page.metadata,
                    "page": index + 1,
                    "total_pages": len(pages),
                    "chunk_method": "smart_pdf_processor",
                },
            )
            for index, page in enumerate(pages)
        ]
        return self._clean_boilerplate(cleaned_pages)

    @staticmethod
    def _clean_text(value: str) -> str:
        value = value.replace("\u00ad", "").replace("\u00a0", " ")
        value = re.sub(r"(?<=\w)-\r?\n(?=\w)", "", value)
        return value.strip()

    @staticmethod
    def _lines(value: str) -> list[str]:
        return [" ".join(line.split()) for line in value.splitlines() if line.strip()]

    def _clean_boilerplate(self, pages: list[Document]) -> list[Document]:
        page_lines = [self._lines(page.page_content) for page in pages]
        edge_counts: Counter[str] = Counter()
        for lines in page_lines:
            edge_counts.update({line.casefold() for line in lines[:3] + lines[-3:]})

        min_repeats = max(2, math.ceil(sum(bool(lines) for lines in page_lines) * 0.6))
        repeated = {line for line, count in edge_counts.items() if count >= min_repeats}

        cleaned_pages = []
        for page, lines in zip(pages, page_lines, strict=True):
            kept_lines = []
            for index, line in enumerate(lines):
                lower = line.casefold()
                near_edge = index < 3 or index >= len(lines) - 3
                if (
                    (near_edge and lower in repeated)
                    or lower.startswith(("home >", "tags:"))
                    or PAGE_FOOTER.fullmatch(line)
                    or PAGE_NUMBER.fullmatch(line)
                ):
                    continue
                kept_lines.append(line)

            if kept_lines:
                content = "\n".join(kept_lines)
                cleaned_pages.append(
                    Document(
                        page_content=content,
                        metadata={**page.metadata, "char_count": len(content)},
                    )
                )
        return cleaned_pages

    @staticmethod
    def _filter_indexable_chunks(chunks: list[Document]) -> list[Document]:
        return [chunk for chunk in chunks if _is_indexable(chunk.page_content)]

    @staticmethod
    def _save_document(
        *,
        file_name: str,
        file_hash: str,
        metadata: DocumentMetadata,
        chunks: list[Document],
        vectors: list[str],
    ) -> int:
        document_values = {
            "file_name": file_name,
            "file_hash": file_hash,
            "title": metadata.title,
            "publisher": metadata.publisher,
            "source_url": str(metadata.source_url),
            "published_at": metadata.published_at,
            "embedding_model": settings.rag_embedding_model,
            "embedding_dimensions": settings.rag_embedding_dimensions,
            "pipeline_version": PIPELINE_VERSION,
        }

        with SessionLocal.begin() as session:
            document_id = session.execute(
                text(
                    """
                    INSERT INTO rag.document (
                        file_name, file_hash, title, publisher, source_url,
                        published_at, embedding_model, embedding_dimensions,
                        pipeline_version
                    ) VALUES (
                        :file_name, :file_hash, :title, :publisher, :source_url,
                        :published_at, :embedding_model, :embedding_dimensions,
                        :pipeline_version
                    )
                    ON CONFLICT (file_name) DO UPDATE SET
                        file_hash = EXCLUDED.file_hash,
                        title = EXCLUDED.title,
                        publisher = EXCLUDED.publisher,
                        source_url = EXCLUDED.source_url,
                        published_at = EXCLUDED.published_at,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dimensions = EXCLUDED.embedding_dimensions,
                        pipeline_version = EXCLUDED.pipeline_version,
                        indexed_at = NOW()
                    RETURNING document_id
                    """
                ),
                document_values,
            ).scalar_one()
            session.execute(
                text("DELETE FROM rag.chunk WHERE document_id = :document_id"),
                {"document_id": document_id},
            )
            session.execute(
                text(
                    """
                    INSERT INTO rag.chunk (
                        document_id, page_number, chunk_index, content, embedding
                    ) VALUES (
                        :document_id, :page_number, :chunk_index, :content,
                        CAST(:embedding AS public.vector)
                    )
                    """
                ),
                [
                    {
                        "document_id": document_id,
                        "page_number": chunk.metadata["page"],
                        "chunk_index": chunk.metadata["chunk_index"],
                        "content": chunk.page_content,
                        "embedding": vector,
                    }
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
        return int(document_id)


class ThresholdSemanticChunker:
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        threshold: float = 0.7,
    ) -> None:
        if not -1 <= threshold <= 1:
            raise ValueError("The cosine similarity threshold must be between -1 and 1.")
        self.model_name = model_name or settings.rag_embedding_model
        self.api_key = api_key
        self.threshold = threshold
        self.long_sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=0
        )

    @cached_property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        api_key = self.api_key
        if api_key is None and settings.gemini_api_key is not None:
            api_key = settings.gemini_api_key.get_secret_value()
        if not api_key or not api_key.strip():
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        return GoogleGenerativeAIEmbeddings(
            model=self.model_name, api_key=api_key.strip(), vertexai=False
        )

    def split(self, text: str) -> list[str]:
        sentences = [
            part
            for sentence in SENTENCE_BOUNDARY.split(text.strip())
            for part in self.long_sentence_splitter.split_text(sentence)
        ]
        if len(sentences) < 2:
            return sentences

        vectors = self.embeddings.embed_documents(
            sentences, task_type="SEMANTIC_SIMILARITY"
        )
        if len(vectors) != len(sentences):
            raise RuntimeError(
                "Gemini returned an unexpected number of sentence vectors."
            )

        chunks = []
        current = sentences[0]
        for sentence, left, right in zip(
            sentences[1:], vectors[:-1], vectors[1:], strict=True
        ):
            combined = f"{current} {sentence}"
            if len(combined) > CHUNK_SIZE or (
                _is_indexable(current)
                and self._cosine_similarity(left, right) < self.threshold
            ):
                chunks.append(current)
                current = sentence
            else:
                current = combined

        if chunks and not _is_indexable(current):
            combined = f"{chunks[-1]} {current}"
            if len(combined) <= CHUNK_SIZE:
                chunks[-1] = combined
                return chunks
        return [*chunks, current]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        magnitude = math.hypot(*left) * math.hypot(*right)
        if not magnitude or not math.isfinite(magnitude):
            raise RuntimeError("Gemini returned an invalid sentence vector.")
        return sum(a * b for a, b in zip(left, right, strict=True)) / magnitude

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return [
            Document(
                page_content=content,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                    "char_count": len(content),
                },
            )
            for document in documents
            for chunk_index, content in enumerate(self.split(document.page_content))
        ]


ThresholdSematicChunker = ThresholdSemanticChunker
