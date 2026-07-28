"""The fixtures are a thin layer, so these tests are thin too.

They run against the installed entry point, which means they also prove the
plugin is actually registered rather than merely importable.
"""

import pytest

from placeborag import FakeEmbedder, FakeVectorStore


def test_fake_embedder_fixture_is_ready_to_use(fake_embedder):
    assert isinstance(fake_embedder, FakeEmbedder)
    assert fake_embedder.embed("refund policy") == fake_embedder.embed("refund policy")


def test_fake_vector_store_fixture_starts_empty(fake_vector_store):
    assert isinstance(fake_vector_store, FakeVectorStore)
    assert len(fake_vector_store) == 0


def test_the_store_uses_the_embedder_fixture(fake_vector_store, fake_embedder):
    fake_vector_store.upsert("doc", "refund policy")

    match = fake_vector_store.query("refund policy", k=1)[0]

    assert match.id == "doc"
    assert fake_embedder.embed("refund policy") == fake_embedder.embed(match.text)


@pytest.mark.placeborag(dimensions=32, model_name="configured")
def test_marker_configures_the_embedder(fake_embedder):
    assert fake_embedder.dimensions == 32
    assert fake_embedder.model_name == "configured"


@pytest.mark.placeborag(clusters={"refund": ["refund policy", "money back"]})
def test_marker_configures_clusters(fake_embedder):
    assert fake_embedder.cluster_names == ["refund"]
    assert fake_embedder.cluster_of("money back") == "refund"


@pytest.mark.placeborag(profile="qdrant", filter_mode="post")
def test_marker_configures_the_store(fake_vector_store):
    assert fake_vector_store.profile.name == "qdrant"
    assert fake_vector_store.filter_mode == "post"


@pytest.mark.placeborag(dimensions=16, profile="qdrant")
def test_marker_options_reach_the_right_object(fake_vector_store):
    # Embedder options and store options come from one marker and must be
    # routed apart.
    assert fake_vector_store.profile.name == "qdrant"
    assert fake_vector_store.embedder.dimensions == 16


def test_unmarked_tests_get_defaults(fake_embedder, fake_vector_store):
    assert fake_embedder.cluster_names == []
    assert fake_vector_store.filter_mode == "pre"


@pytest.mark.placeborag(dimensionz=32)
def test_a_typo_in_the_marker_fails_loudly(request):
    with pytest.raises(pytest.UsageError, match="unknown placeborag marker"):
        request.getfixturevalue("fake_embedder")
