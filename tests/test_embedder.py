import math

import pytest

from placeborag import DEFAULT_DIMENSIONS, FakeEmbedder, cosine_similarity


def test_returns_the_same_vector_for_the_same_text():
    embedder = FakeEmbedder()

    first = embedder.embed("refund policy")
    second = embedder.embed("refund policy")

    assert first == second


def test_two_embedders_with_the_same_config_agree():
    left = FakeEmbedder(model_name="shared", dimensions=64)
    right = FakeEmbedder(model_name="shared", dimensions=64)

    assert left.embed("delivery times") == right.embed("delivery times")


def test_changing_model_name_changes_the_embedding_space():
    text = "refund policy"

    small = FakeEmbedder(model_name="model-a").embed(text)
    other = FakeEmbedder(model_name="model-b").embed(text)

    assert small != other


def test_vector_length_follows_the_dimensions_argument():
    assert len(FakeEmbedder().embed("hello")) == DEFAULT_DIMENSIONS
    assert len(FakeEmbedder(dimensions=16).embed("hello")) == 16


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "!!!",
        "a",
        "refund policy",
        "hogyan kérek vissza pénzt",
        "日本語のテキスト",
        "🙂🙂",
    ],
)
def test_every_vector_is_normalized(text):
    vector = FakeEmbedder(dimensions=32).embed(text)

    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_related_text_scores_higher_than_unrelated_text():
    embedder = FakeEmbedder()
    query = embedder.embed("refund policy")
    related = embedder.embed("what is your refund policy for orders")
    unrelated = embedder.embed("delivery times to remote islands")

    assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)


def test_embed_batch_matches_embed():
    embedder = FakeEmbedder(dimensions=32)
    texts = ["refund policy", "delivery times", ""]

    assert embedder.embed_batch(texts) == [embedder.embed(text) for text in texts]


def test_cosine_similarity_of_a_vector_with_itself_is_one():
    vector = FakeEmbedder(dimensions=32).embed("refund policy")

    assert math.isclose(cosine_similarity(vector, vector), 1.0, rel_tol=1e-9)


def test_cosine_similarity_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_cosine_similarity_rejects_a_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


@pytest.mark.parametrize("dimensions", [0, 1, -8])
def test_rejects_dimensions_below_the_minimum(dimensions):
    with pytest.raises(ValueError, match="dimensions must be"):
        FakeEmbedder(dimensions=dimensions)


def test_rejects_a_non_int_dimensions():
    with pytest.raises(TypeError, match="dimensions must be an int"):
        FakeEmbedder(dimensions=32.0)


def test_rejects_an_empty_model_name():
    with pytest.raises(ValueError, match="model_name must be"):
        FakeEmbedder(model_name="")


def test_rejects_non_string_text():
    with pytest.raises(TypeError, match="text must be a str"):
        FakeEmbedder().embed(None)


def test_repr_shows_the_configuration():
    assert repr(FakeEmbedder(model_name="m", dimensions=8)) == (
        "FakeEmbedder(model_name='m', dimensions=8)"
    )
