from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseRAGProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        return [
            chunk
            for chunk in self.text_splitter.split_documents(documents)
            if chunk.page_content.strip()
        ]

    @staticmethod
    def _require_file(file_path: str, suffixes: set[str]) -> Path:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.casefold() not in suffixes:
            expected = ", ".join(sorted(suffixes))
            raise ValueError(f"Expected {expected}, got: {path.suffix or 'no extension'}")
        return path

    @staticmethod
    def _document(
        content: str,
        path: Path,
        file_type: str,
        **metadata: Any,
    ) -> Document:
        content = content.strip()
        return Document(
            page_content=content,
            metadata={
                "source": str(path),
                "file_name": path.name,
                "file_type": file_type,
                "chunk_method": f"{file_type}_rag_processor",
                "char_count": len(content),
                **metadata,
            },
        )


def row_to_text(headers: list[str], values: Iterable[Any]) -> str:
    fields = []
    for index, value in enumerate(values):
        if value is None or not str(value).strip():
            continue
        header = headers[index] if index < len(headers) else f"column_{index + 1}"
        fields.append(f"{header}: {value}")
    return "\n".join(fields)
