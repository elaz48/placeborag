"""Property-based tests. The guarantees this library sells are property shaped.

The interesting one is `test_arbitrary_declarations_are_satisfiable`: cluster
construction verifies its own geometry and raises when it cannot be satisfied,
so hypothesis failing to trigger that raise is evidence the construction is
robust — not just that it works on the examples we happened to pick.
"""

import itertools
import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from placeborag import FakeEmbedder, FakeVectorStore, cosine_similarity

DIMENSIONS = 64

# Hypothesis loves surrogates and control characters; they are legitimate input
# here, so they stay in.
texts = st.text(max_size=80)
non_empty_texts = st.text(min_size=1, max_size=80)


@given(texts)
def test_embedding_is_deterministic(text):
    embedder = FakeEmbedder(dimensions=DIMENSIONS)

    assert embedder.embed(text) == embedder.embed(text)


@given(texts)
def test_separate_instances_agree(text):
    left = FakeEmbedder(dimensions=DIMENSIONS)
    right = FakeEmbedder(dimensions=DIMENSIONS)

    assert left.embed(text) == right.embed(text)


@given(texts)
def test_every_embedding_is_normalized(text):
    vector = FakeEmbedder(dimensions=DIMENSIONS).embed(text)

    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


@given(texts)
def test_model_name_changes_the_embedding_space(text):
    left = FakeEmbedder(model_name="model-a", dimensions=DIMENSIONS)
    right = FakeEmbedder(model_name="model-b", dimensions=DIMENSIONS)

    assert left.embed(text) != right.embed(text)


@given(texts, st.integers(min_value=2, max_value=512))
def test_dimensions_argument_is_honoured(text, dimensions):
    assert len(FakeEmbedder(dimensions=dimensions).embed(text)) == dimensions


@given(texts)
def test_self_similarity_is_one(text):
    vector = FakeEmbedder(dimensions=DIMENSIONS).embed(text)

    assert cosine_similarity(vector, vector) == pytest.approx(1.0, rel=1e-9)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(non_empty_texts, min_size=2, max_size=4, unique=True),
    st.lists(non_empty_texts, min_size=2, max_size=4, unique=True),
)
def test_arbitrary_declarations_are_satisfiable(left_texts, right_texts):
    assume(not set(left_texts) & set(right_texts))

    embedder = FakeEmbedder(
        dimensions=DIMENSIONS,
        clusters={"left": left_texts, "right": right_texts},
    )

    vectors = {
        text: embedder.embed(text) for text in itertools.chain(left_texts, right_texts)
    }
    intra = [
        cosine_similarity(vectors[a], vectors[b])
        for group in (left_texts, right_texts)
        for a, b in itertools.combinations(group, 2)
    ]
    inter = [
        cosine_similarity(vectors[a], vectors[b])
        for a in left_texts
        for b in right_texts
    ]

    assert min(intra) > max(inter)


@settings(max_examples=50, deadline=None)
@given(st.lists(non_empty_texts, min_size=2, max_size=4, unique=True), texts)
def test_undeclared_text_never_beats_a_declared_sibling(declared, outsider):
    assume(outsider not in declared)

    embedder = FakeEmbedder(dimensions=DIMENSIONS, clusters={"group": declared})
    anchor = embedder.cluster_anchor("group")

    weakest_member = min(
        cosine_similarity(anchor, embedder.embed(text)) for text in declared
    )

    assert weakest_member > cosine_similarity(anchor, embedder.embed(outsider))


@settings(max_examples=50, deadline=None)
@given(st.lists(non_empty_texts, min_size=1, max_size=6, unique=True))
def test_store_returns_at_most_k_and_ranks_stably(documents):
    store = FakeVectorStore(embedder=FakeEmbedder(dimensions=DIMENSIONS))
    for index, text in enumerate(documents):
        store.upsert(f"doc-{index}", text)

    first = [match.id for match in store.query(documents[0], k=3)]
    second = [match.id for match in store.query(documents[0], k=3)]

    assert len(first) <= 3
    assert first == second
