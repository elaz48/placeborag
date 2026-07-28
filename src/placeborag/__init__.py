"""Deterministic test doubles for the retrieval half of a RAG pipeline."""

from placeborag.clusters import DEFAULT_CLUSTER_SPREAD
from placeborag.embedder import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL_NAME,
    FakeEmbedder,
    cosine_similarity,
)
from placeborag.vector_store import (
    CHROMA_PROFILE,
    PROFILES,
    QDRANT_PROFILE,
    BackendProfile,
    FakeVectorStore,
    Match,
)

__all__ = [
    "CHROMA_PROFILE",
    "DEFAULT_CLUSTER_SPREAD",
    "DEFAULT_DIMENSIONS",
    "DEFAULT_MODEL_NAME",
    "PROFILES",
    "QDRANT_PROFILE",
    "BackendProfile",
    "FakeEmbedder",
    "FakeVectorStore",
    "Match",
    "cosine_similarity",
]

__version__ = "0.0.2"
