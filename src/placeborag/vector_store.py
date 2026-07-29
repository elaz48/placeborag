"""A fake vector store that reproduces real backend quirks.

The product here is the quirks, not the correctness. A clean, ideal vector
store catches nothing. Two things differ between real backends and both cause
production bugs that are currently untestable offline:

- **Score conventions.** Chroma-style backends return a distance, where lower
  is better. Qdrant-style backends return a similarity score, where higher is
  better. Picking the wrong one reverses your sort.
- **Filter ordering.** Pre-filtering applies the metadata filter and then takes
  the top k. Post-filtering takes the top k and then filters, so it can
  legitimately return fewer than k results — silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from placeborag.embedder import FakeEmbedder, cosine_similarity
from placeborag.filters import compile_filter

DEFAULT_K = 10

_TIE_BREAK_INSERTION = "insertion"
_TIE_BREAK_ID = "id"


@dataclass(frozen=True)
class BackendProfile:
    """How a backend reports and orders scores.

    Args:
        name: Identifier used in error messages.
        higher_is_better: True if the reported score is a similarity, False if
            it is a distance.
        tie_breaker: Which record wins when two scores are equal. Real stores
            differ here, and the difference is invisible until a test flakes.
    """

    name: str
    higher_is_better: bool
    tie_breaker: str

    def score_for(self, similarity: float) -> float:
        return similarity if self.higher_is_better else 1.0 - similarity


CHROMA_PROFILE = BackendProfile(
    name="chroma", higher_is_better=False, tie_breaker=_TIE_BREAK_INSERTION
)
QDRANT_PROFILE = BackendProfile(
    name="qdrant", higher_is_better=True, tie_breaker=_TIE_BREAK_ID
)

PROFILES: Mapping[str, BackendProfile] = {
    CHROMA_PROFILE.name: CHROMA_PROFILE,
    QDRANT_PROFILE.name: QDRANT_PROFILE,
}

FILTER_MODES = ("pre", "post")


@dataclass(frozen=True)
class Match:
    """One query result.

    `score` follows the store's profile: a distance when the profile reports
    distances, a similarity when it reports similarities. Check
    `store.profile.higher_is_better` before comparing it against a threshold.
    """

    id: str
    text: str
    metadata: Mapping[str, Any]
    score: float


@dataclass(frozen=True)
class _Record:
    id: str
    text: str
    metadata: Mapping[str, Any]
    vector: list[float] = field(repr=False)
    sequence: int


class FakeVectorStore:
    """An in-memory vector store with a configurable backend personality."""

    def __init__(
        self,
        embedder: FakeEmbedder | None = None,
        profile: str | BackendProfile = CHROMA_PROFILE.name,
        filter_mode: str = "pre",
    ) -> None:
        if filter_mode not in FILTER_MODES:
            raise ValueError(
                f"filter_mode must be one of {FILTER_MODES}, got {filter_mode!r}"
            )

        self._profile = _resolve_profile(profile)
        self._filter_mode = filter_mode
        self._embedder = embedder if embedder is not None else FakeEmbedder()
        self._records: dict[str, _Record] = {}
        self._next_sequence = 0

    @property
    def embedder(self) -> FakeEmbedder:
        """The embedder this store indexes and queries with."""
        return self._embedder

    @property
    def profile(self) -> BackendProfile:
        return self._profile

    @property
    def filter_mode(self) -> str:
        return self._filter_mode

    @property
    def ids(self) -> list[str]:
        """Record ids in insertion order."""
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def upsert(
        self,
        id: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Adds a record, or replaces the one already stored under `id`.

        Replacing a record keeps its original position in insertion order,
        which is what decides ties under an insertion-order profile.
        """
        if not isinstance(id, str) or not id:
            raise ValueError("id must be a non-empty string")
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")

        existing = self._records.get(id)
        sequence = existing.sequence if existing else self._take_sequence()
        self._records[id] = _Record(
            id=id,
            text=text,
            metadata=dict(metadata or {}),
            vector=self._embedder.embed(text),
            sequence=sequence,
        )

    def delete(self, id: str) -> bool:
        """Removes a record. Returns False if there was nothing to remove."""
        return self._records.pop(id, None) is not None

    def query(
        self,
        text: str,
        k: int = DEFAULT_K,
        where: Mapping[str, Any] | None = None,
    ) -> list[Match]:
        """Returns up to `k` matches, best first.

        `where` supports equality (`{"lang": "hu"}`) and operators
        (`{"year": {"$gte": 2024}}`, `{"lang": {"$in": [...]}}`, `$and`,
        `$or`). A malformed clause raises here, before any record is read.

        Under `filter_mode="post"` this can return fewer than `k` matches even
        when more records satisfy `where`, because the filter runs after the
        top-k cut. That is the behaviour of real post-filtering backends and
        the reason this store exists.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("k must be an int")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        matches_filter = compile_filter(where)
        query_vector = self._embedder.embed(text)
        candidates = list(self._records.values())
        if self._filter_mode == "pre" and where:
            candidates = [
                record for record in candidates if matches_filter(record.metadata)
            ]

        ranked = self._rank(candidates, query_vector)[:k]
        if self._filter_mode == "post" and where:
            ranked = [record for record in ranked if matches_filter(record.metadata)]

        return [self._as_match(record, query_vector) for record in ranked]

    def _rank(self, records: list[_Record], query_vector: list[float]) -> list[_Record]:
        tie_break = (
            (lambda record: record.sequence)
            if self._profile.tie_breaker == _TIE_BREAK_INSERTION
            else (lambda record: record.id)
        )
        # Relevance order is the same under both conventions; only the reported
        # score differs. Sorting on similarity keeps that explicit.
        return sorted(
            records,
            key=lambda record: (
                -cosine_similarity(query_vector, record.vector),
                tie_break(record),
            ),
        )

    def _as_match(self, record: _Record, query_vector: list[float]) -> Match:
        similarity = cosine_similarity(query_vector, record.vector)
        return Match(
            id=record.id,
            text=record.text,
            metadata=dict(record.metadata),
            score=self._profile.score_for(similarity),
        )

    def _take_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        return sequence

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(profile={self._profile.name!r}, "
            f"filter_mode={self._filter_mode!r}, records={len(self._records)})"
        )


def _resolve_profile(profile: str | BackendProfile) -> BackendProfile:
    if isinstance(profile, BackendProfile):
        return profile
    try:
        return PROFILES[profile]
    except (KeyError, TypeError):
        available = ", ".join(sorted(PROFILES))
        raise ValueError(
            f"unknown profile {profile!r}, available profiles: {available}"
        ) from None
