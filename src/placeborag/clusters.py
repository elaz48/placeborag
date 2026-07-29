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

MATCH_EXACT = "exact"
MATCH_SUBSTRING = "substring"
MATCH_MODES = (MATCH_EXACT, MATCH_SUBSTRING)
DEFAULT_CLUSTER_MATCH = MATCH_EXACT

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
        match: str = DEFAULT_CLUSTER_MATCH,
    ) -> None:
        _validate_declaration(clusters)
        self._key = key
        self._magnitude_key = magnitude_key
        self._dimensions = dimensions
        self._spread = spread
        self._match = validate_match(match)
        self._anchors, self._vectors = _build_vectors(
            clusters, key, magnitude_key, dimensions, spread
        )
        self._cluster_of = {
            text: name for name, texts in clusters.items() for text in texts
        }
        # Longest declaration first, so the most specific one wins when a
        # chunk contains several.
        self._by_specificity = sorted(self._cluster_of, key=len, reverse=True)
        _assert_separable(clusters, self._vectors)

    def vector_for(self, text: str) -> list[float] | None:
        """The vector for an exactly declared text, or None."""
        vector = self._vectors.get(text)
        return list(vector) if vector is not None else None

    def derive(self, text: str, hashed: Sequence[float]) -> list[float] | None:
        """Pulls a text containing a declared member into that cluster.

        Only active in `substring` mode. The offset from the anchor is the
        text's own hashing vector, orthogonalized against that anchor — so
        two chunks that hash near each other stay near each other *inside*
        the cluster, instead of being scattered by unrelated jitter. The
        cosine to the anchor is still exactly `1 / sqrt(1 + magnitude**2)`,
        so the dimension-independence guarantee is unaffected.
        """
        if self._match != MATCH_SUBSTRING:
            return None
        member = self._matching_member(text)
        if member is None:
            return None

        name = self._cluster_of[member]
        anchor = self._anchors[name]
        offset = _orthogonal_component(hashed, anchor)
        if offset is None:
            # The hashing vector is parallel to the anchor, so it carries no
            # usable offset. Fall back to a generated direction.
            offset = _orthogonal_unit_vector(
                self._key, f"jitter:{name}:{text}", anchor, self._dimensions
            )
        magnitude = _jitter_magnitude(
            self._magnitude_key, f"magnitude:{name}:{text}", self._spread
        )
        return list(_combine(anchor, offset, magnitude))

    def _matching_member(self, text: str) -> str | None:
        return next((member for member in self._by_specificity if member in text), None)

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
        name = self._cluster_of.get(text)
        if name is not None or self._match == MATCH_EXACT:
            return name
        member = self._matching_member(text)
        return self._cluster_of[member] if member is not None else None

    @property
    def names(self) -> list[str]:
        return sorted(set(self._cluster_of.values()))


def validate_match(match: str) -> str:
    if match not in MATCH_MODES:
        raise ValueError(f"cluster_match must be one of {MATCH_MODES}, got {match!r}")
    return match


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
            "clusters must be a mapping of name to texts, "
            f"got {type(clusters).__name__}"
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
    combined = [a + magnitude * j for a, j in zip(anchor, jitter, strict=False)]
    norm = math.sqrt(sum(value * value for value in combined))
    return tuple(value / norm for value in combined)


def _jitter_magnitude(key: bytes, label: str, spread: float) -> float:
    """Magnitude depends only on the text, never on `dimensions`.

    This is what keeps within-cluster ordering stable across dimensions: the
    cosine to the anchor is 1 / sqrt(1 + magnitude**2), with no dimension term.
    """
    return spread * (0.5 + _uniform_floats(key, label, 1)[0])


def _orthogonal_component(
    vector: Sequence[float], anchor: tuple[float, ...]
) -> tuple[float, ...] | None:
    """The part of `vector` perpendicular to `anchor`, normalized.

    None when `vector` is (numerically) parallel to the anchor and therefore
    has no perpendicular part to speak of.
    """
    projection = sum(v * a for v, a in zip(vector, anchor, strict=False))
    residual = [v - projection * a for v, a in zip(vector, anchor, strict=False)]
    norm = math.sqrt(sum(value * value for value in residual))
    if norm <= _MIN_USABLE_NORM:
        return None
    return tuple(value / norm for value in residual)


def _orthogonal_unit_vector(
    key: bytes, label: str, anchor: tuple[float, ...], dimensions: int
) -> tuple[float, ...]:
    for attempt in range(_ORTHOGONALIZATION_ATTEMPTS):
        candidate = _unit_vector(key, f"{label}#{attempt}", dimensions)
        projection = sum(c * a for c, a in zip(candidate, anchor, strict=False))
        residual = [c - projection * a for c, a in zip(candidate, anchor, strict=False)]
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
    return sum(a * b for a, b in zip(left, right, strict=False))
