"""A deterministic embedder built on feature hashing.

No model, no data file, no network. `embed` is a pure function of
`(text, model_name, dimensions)`, so the same text always produces the same
vector, and two texts that share tokens land near each other. That is what
makes a top-k assertion in a test mean something.
"""

from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections.abc import Iterable, Mapping, Sequence

from placeborag.clusters import (
    DEFAULT_CLUSTER_SPREAD,
    ClusterSpace,
    validate_spread,
)

DEFAULT_MODEL_NAME = "placebo-hash-001"
DEFAULT_DIMENSIONS = 256

_MIN_DIMENSIONS = 2
_CHAR_NGRAM_SIZES = (3, 4)
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_KEY_SIZE = 32
_SIGN_BIT = 0x80000000
_SEED_BYTES = 4
# Each feature category gets its own seed, which keeps a word token and an
# identical-looking n-gram in different buckets without paying for a string
# prefix on every feature in the hot loop.
_WORD_SEED_INDEX = 0
_SENTINEL_SEED_INDEX = 1
_NGRAM_SEED_OFFSET = 2


class FakeEmbedder:
    """Embeds text into a deterministic unit vector.

    Args:
        model_name: Salts the hash. A different name yields a different but
            equally deterministic embedding space, which is what makes the
            "we swapped the embedding model and have to reindex" code path
            testable.
        dimensions: Length of the returned vector.
        clusters: Optional mapping of cluster name to the texts that belong to
            it. Declared texts get a vector near their cluster's anchor;
            everything else falls through to the hashing layer. The
            declaration is checked at construction time and raises if it
            cannot be satisfied.
        cluster_spread: How far members sit from their anchor. Smaller means
            tighter clusters.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dimensions: int = DEFAULT_DIMENSIONS,
        clusters: Mapping[str, Sequence[str]] | None = None,
        cluster_spread: float = DEFAULT_CLUSTER_SPREAD,
    ) -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool):
            raise TypeError("dimensions must be an int")
        if dimensions < _MIN_DIMENSIONS:
            raise ValueError(f"dimensions must be >= {_MIN_DIMENSIONS}, got {dimensions}")

        self._model_name = model_name
        self._dimensions = dimensions
        self._cluster_spread = validate_spread(cluster_spread)
        self._key = hashlib.blake2b(
            f"{model_name}:{dimensions}".encode(), digest_size=_KEY_SIZE
        ).digest()
        # Same model name, any dimensions: used for quantities that must not
        # vary with the vector length.
        self._model_key = hashlib.blake2b(
            model_name.encode(), digest_size=_KEY_SIZE
        ).digest()
        self._word_seed = _seed(self._key, _WORD_SEED_INDEX)
        self._sentinel_seed = _seed(self._key, _SENTINEL_SEED_INDEX)
        self._ngram_seeds = tuple(
            (size, _seed(self._key, _NGRAM_SEED_OFFSET + offset))
            for offset, size in enumerate(_CHAR_NGRAM_SIZES)
        )
        self._clusters = (
            ClusterSpace(
                clusters,
                key=self._key,
                magnitude_key=self._model_key,
                dimensions=dimensions,
                spread=self._cluster_spread,
            )
            if clusters is not None
            else None
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def cluster_names(self) -> list[str]:
        """Names of the declared clusters, empty when none were declared."""
        return self._clusters.names if self._clusters else []

    def cluster_of(self, text: str) -> str | None:
        """The cluster `text` was declared in, or None if it was not."""
        return self._clusters.cluster_of(text) if self._clusters else None

    def cluster_anchor(self, name: str) -> list[float] | None:
        """The centre of a declared cluster, or None if there is no such cluster.

        Use it as a query vector to rank a cluster's members in an order that
        does not depend on `dimensions`.
        """
        return self._clusters.anchor_for(name) if self._clusters else None

    def embed(self, text: str) -> list[float]:
        """Returns the L2-normalized vector for `text`.

        Declared cluster members get their cluster vector. Everything else is
        hashed.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")

        if self._clusters is not None:
            declared = self._clusters.vector_for(text)
            if declared is not None:
                return declared

        vector = self._accumulate(text)
        norm = _norm(vector)
        if norm == 0.0:
            # Signs cancelled out exactly. Vanishingly unlikely, but a zero
            # vector cannot be normalized, so fall back to the sentinel.
            vector = [0.0] * self._dimensions
            self._add_feature(vector, _sentinel(text), self._sentinel_seed)
            norm = _norm(vector)
        return [value / norm for value in vector]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Returns one vector per input text, in order."""
        return [self.embed(text) for text in texts]

    def _accumulate(self, text: str) -> list[float]:
        """Sums the signed hash of every feature into the bucket vector.

        Kept as one flat loop on purpose: this runs once per feature, and
        there are roughly two features per character of input.
        """
        dimensions = self._dimensions
        vector = [0.0] * dimensions
        lowered = text.lower()
        found = False

        word_seed = self._word_seed
        for token in _WORD_PATTERN.findall(lowered):
            value = zlib.crc32(token.encode("utf-8"), word_seed)
            vector[value % dimensions] += -1.0 if value & _SIGN_BIT else 1.0
            found = True

        for size, seed in self._ngram_seeds:
            for start in range(len(lowered) - size + 1):
                value = zlib.crc32(lowered[start : start + size].encode("utf-8"), seed)
                vector[value % dimensions] += -1.0 if value & _SIGN_BIT else 1.0
                found = True

        if not found:
            # No word tokens and nothing long enough for an n-gram: the empty
            # string, a single character, punctuation only.
            self._add_feature(vector, _sentinel(text), self._sentinel_seed)
        return vector

    def _add_feature(self, vector: list[float], feature: str, seed: int) -> None:
        value = zlib.crc32(feature.encode("utf-8"), seed)
        vector[value % self._dimensions] += -1.0 if value & _SIGN_BIT else 1.0

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_name={self._model_name!r}, "
            f"dimensions={self._dimensions})"
        )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two vectors of equal length."""
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} != {len(right)}")
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    dot = sum(a * b for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def _seed(key: bytes, index: int) -> int:
    """A 32-bit crc32 seed carved out of the model key."""
    start = index * _SEED_BYTES
    return int.from_bytes(key[start : start + _SEED_BYTES], "big")


def _sentinel(text: str) -> str:
    """Feature used for inputs that produce no n-grams or word tokens."""
    return f"\x00empty\x00{text}"


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))
