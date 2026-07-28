"""pytest fixtures, as a thin layer over the library.

The library is the product; this is convenience. Everything here is reachable
without pytest, and nothing here holds state the library does not.

Configure per test with the `placeborag` marker:

    @pytest.mark.placeborag(clusters={"refund": ["refund policy", "money back"]})
    def test_retrieval(fake_vector_store): ...
"""

from __future__ import annotations

from typing import Any

import pytest

from placeborag.vector_store import FakeEmbedder, FakeVectorStore

MARKER_NAME = "placeborag"

_EMBEDDER_OPTIONS = frozenset(
    {"model_name", "dimensions", "clusters", "cluster_spread"}
)
_STORE_OPTIONS = frozenset({"profile", "filter_mode"})


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER_NAME}(**options): configure the placeborag fixtures. "
        f"Embedder options: {', '.join(sorted(_EMBEDDER_OPTIONS))}. "
        f"Store options: {', '.join(sorted(_STORE_OPTIONS))}.",
    )


@pytest.fixture
def placeborag_options(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Options from the closest `placeborag` marker, validated.

    An unknown option is an error rather than a silently ignored typo — a
    marker that does nothing is worse than one that fails.
    """
    marker = request.node.get_closest_marker(MARKER_NAME)
    options = dict(marker.kwargs) if marker else {}

    unknown = set(options) - _EMBEDDER_OPTIONS - _STORE_OPTIONS
    if unknown:
        known = sorted(_EMBEDDER_OPTIONS | _STORE_OPTIONS)
        raise pytest.UsageError(
            f"unknown {MARKER_NAME} marker option(s): {', '.join(sorted(unknown))}. "
            f"Known options: {', '.join(known)}"
        )
    return options


@pytest.fixture
def fake_embedder(placeborag_options: dict[str, Any]) -> FakeEmbedder:
    """A deterministic embedder, configurable via the `placeborag` marker."""
    return FakeEmbedder(
        **{
            name: value
            for name, value in placeborag_options.items()
            if name in _EMBEDDER_OPTIONS
        }
    )


@pytest.fixture
def fake_vector_store(
    fake_embedder: FakeEmbedder, placeborag_options: dict[str, Any]
) -> FakeVectorStore:
    """An empty vector store wired to `fake_embedder`."""
    return FakeVectorStore(
        embedder=fake_embedder,
        **{
            name: value
            for name, value in placeborag_options.items()
            if name in _STORE_OPTIONS
        },
    )
