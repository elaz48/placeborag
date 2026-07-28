"""Declared clusters: the control layer over the hashing base.

Each cluster gets a seeded anchor on the unit sphere. Members are the anchor
plus a deterministic jitter that is **orthogonal to the anchor**, which makes
the cosine between a member and its anchor exactly `1 / sqrt(1 + m**2)` for
jitter magnitude `m`. That identity has no dimension term in it, which is why
ordering within a cluster survives a change of `dimensions`.

The declaration is checked at construction time and violations raise. A
retrieval test should be a statement of intent, not a lottery.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Mapping, Sequence

DEFAULT_CLUSTER_SPREAD = 0.15

_MAX_CLUSTER_SPREAD = 1.0
_UNIFORM_DIVISOR = float(1 << 64)
_BYTES_PER_FLOAT = 8
_DIGEST_SIZE = 64
_ORTHOGONALIZATION_ATTEMPTS = 8
_MIN_USABLE_NORM = 1e-9


class ClusterSpace:
    """Deterministic vectors for the texts a user declared."""

    def __init__(
        self,
        clusters: Mapping[str, Sequence[str]],
        key: bytes,
        magnitude_key: bytes,
        dimensions: int,
        spread: float,
    ) -> None:
        _validate_declaration(clusters)
        self._spread = spread
        self._anchors, self._vectors = _build_vectors(
            clusters, key, magnitude_key, dimensions, spread
        )
        self._cluster_of = {
            text: name for name, texts in clusters.items() for text in texts
        }
        _assert_separable(clusters, self._vectors)

    def vector_for(self, text: str) -> list[float] | None:
        """Returns a copy of the declared vector, or None if undeclared."""
        vector = self._vectors.get(text)
        return list(vector) if vector is not None else None

    def anchor_for(self, name: str) -> list[float] | None:
        """The cluster's centre. Querying with it ranks members by jitter.

        The cosine between an anchor and one of its members is exactly
        `1 / sqrt(1 + magnitude**2)`, so ranking members against their anchor
        is identical at every `dimensions`. Ranking members against *each
        other* is not: that involves the angle between two jitter directions,
        which does depend on the dimension.
        """
        anchor = self._anchors.get(name)
        return list(anchor) if anchor is not None else None

    def cluster_of(self, text: str) -> str | None:
        return self._cluster_of.get(text)

    @property
    def names(self) -> list[str]:
        return sorted(set(self._cluster_of.values()))


def validate_spread(spread: float) -> float:
    if isinstance(spread, bool) or not isinstance(spread, (int, float)):
        raise TypeError("cluster_spread must be a number")
    if not 0.0 < spread <= _MAX_CLUSTER_SPREAD:
        raise ValueError(
            f"cluster_spread must be in (0, {_MAX_CLUSTER_SPREAD}], got {spread}"
        )
    return float(spread)


def _validate_declaration(clusters: Mapping[str, Sequence[str]]) -> None:
    if not isinstance(clusters, Mapping):
        raise TypeError(
            f"clusters must be a mapping of name to texts, got {type(clusters).__name__}"
        )

    seen: dict[str, str] = {}
    for name, texts in clusters.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"cluster name must be a non-empty string, got {name!r}")
        # A bare string is a Sequence[str] of characters, which would silently
        # cluster every letter. Reject it explicitly.
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise TypeError(
                f"cluster member list for {name!r} must be a list of strings, "
                f"got {type(texts).__name__}"
            )
        if not texts:
            raise ValueError(f"cluster {name!r} must declare at least one member")

        for text in texts:
            if not isinstance(text, str):
                raise TypeError(
                    f"cluster member in {name!r} must be a str, "
                    f"got {type(text).__name__}"
                )
            previous = seen.get(text)
            if previous is not None and previous != name:
                raise ValueError(
                    f"{text!r} is declared in more than one cluster "
                    f"({previous!r} and {name!r}); a text can belong to one cluster"
                )
            seen[text] = name


def _build_vectors(
    clusters: Mapping[str, Sequence[str]],
    key: bytes,
    magnitude_key: bytes,
    dimensions: int,
    spread: float,
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    anchors: dict[str, tuple[float, ...]] = {}
    vectors: dict[str, tuple[float, ...]] = {}
    for name, texts in clusters.items():
        anchor = _unit_vector(key, f"anchor:{name}", dimensions)
        anchors[name] = anchor
        for text in texts:
            jitter = _orthogonal_unit_vector(
                key, f"jitter:{name}:{text}", anchor, dimensions
            )
            # Keyed without the dimension, so the cosine to the anchor is the
            # same number at every vector length.
            magnitude = _jitter_magnitude(
                magnitude_key, f"magnitude:{name}:{text}", spread
            )
            vectors[text] = _combine(anchor, jitter, magnitude)
    return anchors, vectors


def _assert_separable(
    clusters: Mapping[str, Sequence[str]],
    vectors: Mapping[str, tuple[float, ...]],
) -> None:
    """Fails loudly if the declaration does not hold geometrically.

    Everything here is deterministic, so a declaration that passes this check
    passes it on every machine and every run. What it cannot check is texts the
    user never declared: those go through the hashing layer, where separation
    is overwhelmingly likely but not proven.
    """
    intra = [
        _cosine(vectors[a], vectors[b])
        for texts in clusters.values()
        for a, b in itertools.combinations(texts, 2)
    ]
    inter = [
        _cosine(vectors[a], vectors[b])
        for left, right in itertools.combinations(clusters, 2)
        for a in clusters[left]
        for b in clusters[right]
    ]
    if not intra or not inter:
        return

    weakest_pair, strongest_leak = min(intra), max(inter)
    if weakest_pair <= strongest_leak:
        raise ValueError(
            "cluster declaration cannot be satisfied: the weakest within-cluster "
            f"similarity ({weakest_pair:.4f}) does not exceed the strongest "
            f"across-cluster similarity ({strongest_leak:.4f}). Try a smaller "
            "cluster_spread, more dimensions, or fewer clusters."
        )


def _combine(
    anchor: tuple[float, ...], jitter: tuple[float, ...], magnitude: float
) -> tuple[float, ...]:
    combined = [a + magnitude * j for a, j in zip(anchor, jitter)]
    norm = math.sqrt(sum(value * value for value in combined))
    return tuple(value / norm for value in combined)


def _jitter_magnitude(key: bytes, label: str, spread: float) -> float:
    """Magnitude depends only on the text, never on `dimensions`.

    This is what keeps within-cluster ordering stable across dimensions: the
    cosine to the anchor is 1 / sqrt(1 + magnitude**2), with no dimension term.
    """
    return spread * (0.5 + _uniform_floats(key, label, 1)[0])


def _orthogonal_unit_vector(
    key: bytes, label: str, anchor: tuple[float, ...], dimensions: int
) -> tuple[float, ...]:
    for attempt in range(_ORTHOGONALIZATION_ATTEMPTS):
        candidate = _unit_vector(key, f"{label}#{attempt}", dimensions)
        projection = sum(c * a for c, a in zip(candidate, anchor))
        residual = [c - projection * a for c, a in zip(candidate, anchor)]
        norm = math.sqrt(sum(value * value for value in residual))
        if norm > _MIN_USABLE_NORM:
            return tuple(value / norm for value in residual)
    raise RuntimeError(  # pragma: no cover - needs a pathological hash collision
        f"could not find a direction orthogonal to the anchor for {label!r}"
    )


def _unit_vector(key: bytes, label: str, dimensions: int) -> tuple[float, ...]:
    """A deterministic point on the unit sphere, Box-Muller from a hash stream."""
    for attempt in range(_ORTHOGONALIZATION_ATTEMPTS):
        pairs = (dimensions + 1) // 2
        uniforms = _uniform_floats(key, f"{label}${attempt}", pairs * 2)
        values: list[float] = []
        for index in range(pairs):
            radius = math.sqrt(-2.0 * math.log(uniforms[2 * index]))
            angle = 2.0 * math.pi * uniforms[2 * index + 1]
            values.append(radius * math.cos(angle))
            values.append(radius * math.sin(angle))

        vector = values[:dimensions]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > _MIN_USABLE_NORM:
            return tuple(value / norm for value in vector)
    raise RuntimeError(  # pragma: no cover - needs a pathological hash collision
        f"could not derive a unit vector for {label!r}"
    )


def _uniform_floats(key: bytes, label: str, count: int) -> list[float]:
    """Deterministic floats in the open interval (0, 1)."""
    values: list[float] = []
    block = 0
    while len(values) < count:
        digest = hashlib.blake2b(
            f"{label}#{block}".encode(), key=key, digest_size=_DIGEST_SIZE
        ).digest()
        for offset in range(0, _DIGEST_SIZE, _BYTES_PER_FLOAT):
            if len(values) == count:
                break
            raw = int.from_bytes(digest[offset : offset + _BYTES_PER_FLOAT], "big")
            # +0.5 keeps this off both endpoints; log(0) would blow up Box-Muller.
            values.append((raw + 0.5) / _UNIFORM_DIVISOR)
        block += 1
    return values


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))
