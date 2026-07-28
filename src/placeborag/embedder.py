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
from collections.abc import Iterable, Sequence

DEFAULT_MODEL_NAME = "placebo-hash-001"
DEFAULT_DIMENSIONS = 256

_MIN_DIMENSIONS = 2
_CHAR_NGRAM_SIZES = (3, 4)
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_KEY_SIZE = 32
_DIGEST_SIZE = 8
_SIGN_BIT = 1 << (_DIGEST_SIZE * 8 - 1)


class FakeEmbedder:
    """Embeds text into a deterministic unit vector.

    Args:
        model_name: Salts the hash. A different name yields a different but
            equally deterministic embedding space, which is what makes the
            "we swapped the embedding model and have to reindex" code path
            testable.
        dimensions: Length of the returned vector.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dimensions: int = DEFAULT_DIMENSIONS,
    ) -> None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(dimensions, int) or isinstance(dimensions, bool):
            raise TypeError("dimensions must be an int")
        if dimensions < _MIN_DIMENSIONS:
            raise ValueError(f"dimensions must be >= {_MIN_DIMENSIONS}, got {dimensions}")

        self._model_name = model_name
        self._dimensions = dimensions
        self._key = hashlib.blake2b(
            f"{model_name}:{dimensions}".encode(), digest_size=_KEY_SIZE
        ).digest()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Returns the L2-normalized vector for `text`."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")

        features = _features(text) or [_sentinel(text)]
        vector = self._accumulate(features)
        norm = _norm(vector)
        if norm == 0.0:
            # Signs cancelled out exactly. Vanishingly unlikely, but a zero
            # vector cannot be normalized, so fall back to the sentinel.
            vector = self._accumulate([_sentinel(text)])
            norm = _norm(vector)
        return [value / norm for value in vector]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Returns one vector per input text, in order."""
        return [self.embed(text) for text in texts]

    def _accumulate(self, features: Sequence[str]) -> list[float]:
        vector = [0.0] * self._dimensions
        for feature in features:
            index, sign = self._bucket(feature)
            vector[index] += sign
        return vector

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(
            feature.encode("utf-8"), key=self._key, digest_size=_DIGEST_SIZE
        ).digest()
        value = int.from_bytes(digest, "big")
        index = value % self._dimensions
        sign = -1.0 if value & _SIGN_BIT else 1.0
        return index, sign

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


def _features(text: str) -> list[str]:
    lowered = text.lower()
    features = [f"w:{token}" for token in _WORD_PATTERN.findall(lowered)]
    for size in _CHAR_NGRAM_SIZES:
        features.extend(
            f"c{size}:{lowered[start : start + size]}"
            for start in range(len(lowered) - size + 1)
        )
    return features


def _sentinel(text: str) -> str:
    """Feature used for inputs that produce no n-grams or word tokens."""
    return f"\x00empty\x00{text}"


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))
