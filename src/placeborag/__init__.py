"""Deterministic test doubles for the retrieval half of a RAG pipeline."""

from placeborag.embedder import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL_NAME,
    FakeEmbedder,
    cosine_similarity,
)

__all__ = [
    "DEFAULT_DIMENSIONS",
    "DEFAULT_MODEL_NAME",
    "FakeEmbedder",
    "cosine_similarity",
]

__version__ = "0.0.1"
