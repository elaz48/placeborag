"""A small RAG pipeline, written the way you would write a real one.

This is the code under test — it knows nothing about placeborag. The embedder
and the vector store arrive by injection, which is the only thing a pipeline
has to do to become testable offline.

Run the accompanying tests with `pytest examples/`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

CHUNK_SIZE_WORDS = 12
CHUNK_OVERLAP_WORDS = 4
DEFAULT_TOP_K = 3


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def upsert(
        self, id: str, text: str, metadata: dict[str, Any] | None = None
    ) -> None: ...

    def query(
        self, text: str, k: int = ..., where: dict[str, Any] | None = None
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    lang: str = "en"
    tier: str = "public"


@dataclass(frozen=True)
class Answer:
    text: str
    sources: tuple[str, ...]


def chunk(text: str, size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    """Splits into overlapping word windows. Deliberately naive."""
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]

    step = size - overlap
    return [
        " ".join(words[start : start + size]) for start in range(0, len(words), step)
    ][: -1 if len(words) % step else None] or [" ".join(words[:size])]


class SupportBot:
    """Indexes support documents and answers questions from them."""

    def __init__(
        self,
        store: VectorStore,
        generate: Callable[[str, Sequence[str]], str],
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._store = store
        self._generate = generate
        self._top_k = top_k

    def index(self, documents: Iterable[Document]) -> int:
        """Chunks and indexes documents. Returns the number of chunks written."""
        written = 0
        for document in documents:
            for position, piece in enumerate(chunk(document.text)):
                self._store.upsert(
                    f"{document.id}#{position}",
                    piece,
                    metadata={
                        "document_id": document.id,
                        "lang": document.lang,
                        "tier": document.tier,
                    },
                )
                written += 1
        return written

    def answer(self, question: str, lang: str | None = None) -> Answer:
        where = {"lang": lang} if lang else None
        matches = self._store.query(question, k=self._top_k, where=where)
        passages = [match.text for match in matches]
        sources = tuple(dict.fromkeys(match.metadata["document_id"] for match in matches))
        return Answer(text=self._generate(question, passages), sources=sources)
