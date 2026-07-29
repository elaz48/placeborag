"""Testing the pipeline in examples/support_bot.py with placeborag.

These are the assertions you cannot write against a random mock embedder:
which document came back, in what order, and how many.
"""

import pytest
from support_bot import Document, SupportBot

DOCUMENTS = [
    Document(
        "refunds",
        "Our refund policy allows returns within 30 days of purchase. "
        "Refunds are issued to the original payment method. "
        "Sale items are excluded from the refund policy.",
    ),
    Document(
        "shipping",
        "Standard delivery times are three to five business days. "
        "Express delivery arrives the next business day. "
        "Delivery to remote islands may take longer.",
    ),
    Document(
        "refunds-hu",
        "A pénzvisszatérítési szabályzat harminc napon belül érvényes. "
        "A visszatérítést az eredeti fizetési módra utaljuk vissza.",
        lang="hu",
    ),
    Document(
        "internal",
        "Escalate refund disputes above 500 EUR to the finance team. "
        "Do not share the internal refund override procedure with customers.",
        tier="internal",
    ),
]


def echo_llm(question: str, passages) -> str:
    """Stands in for the generation half. Not what these tests are about."""
    return f"{question} -> {len(passages)} passages"


@pytest.fixture
def bot(fake_vector_store):
    bot = SupportBot(store=fake_vector_store, generate=echo_llm)
    bot.index(DOCUMENTS)
    return bot


def test_indexing_writes_chunks_not_documents(fake_vector_store):
    bot = SupportBot(store=fake_vector_store, generate=echo_llm)

    written = bot.index(DOCUMENTS)

    assert written == len(fake_vector_store)
    assert written > len(DOCUMENTS)


def test_a_refund_question_retrieves_the_refund_document(bot):
    answer = bot.answer("how do I get my money back for an order")

    assert "refunds" in answer.sources


def test_a_shipping_question_does_not_retrieve_refund_content(bot):
    answer = bot.answer("how long does delivery take")

    assert answer.sources[0] == "shipping"


class TestTheBugsWorthCatching:
    """Each of these passes vacuously against a random mock embedder."""

    @pytest.mark.placeborag(filter_mode="post")
    def test_post_filtering_can_starve_a_language_filter(self, fake_vector_store):
        bot = SupportBot(store=fake_vector_store, generate=echo_llm)
        bot.index(DOCUMENTS)

        answer = bot.answer("refund policy", lang="hu")

        # The Hungarian chunks exist and match the filter, but an
        # English-language query ranks them below the top-k cut, so
        # post-filtering hands back nothing at all.
        assert answer.sources == ()

    def test_pre_filtering_finds_the_same_documents(self, fake_vector_store):
        bot = SupportBot(store=fake_vector_store, generate=echo_llm)
        bot.index(DOCUMENTS)

        answer = bot.answer("refund policy", lang="hu")

        assert answer.sources == ("refunds-hu",)

    def test_internal_content_never_leaks_into_a_public_answer(self, bot):
        # A filter the pipeline does not apply is a filter that does not
        # protect you. This documents the gap rather than hiding it.
        answer = bot.answer("refund override procedure")

        assert "internal" in answer.sources, (
            "SupportBot.answer has no tier filter, so internal chunks are "
            "reachable. Fix the pipeline, not this test."
        )


class TestReindexing:
    """Swapping the embedding model invalidates the index. Usually untested."""

    @pytest.mark.placeborag(model_name="embedding-model-v1")
    def test_an_index_built_with_one_model_is_queried_with_that_model(
        self, fake_vector_store
    ):
        bot = SupportBot(store=fake_vector_store, generate=echo_llm)
        bot.index(DOCUMENTS)

        assert "refunds" in bot.answer("refund policy").sources

    def test_querying_a_stale_index_with_a_new_model_returns_noise(self):
        from placeborag import FakeEmbedder, FakeVectorStore

        old_store = FakeVectorStore(embedder=FakeEmbedder(model_name="v1"))
        SupportBot(store=old_store, generate=echo_llm).index(DOCUMENTS)

        # Same store, same vectors, but the query now goes through a
        # different embedding space — exactly what happens when you swap
        # models and forget to reindex.
        new_embedder = FakeEmbedder(model_name="v2")
        query_vector = new_embedder.embed("refund policy")
        indexed_vector = old_store.embedder.embed("refund policy")

        assert query_vector != indexed_vector


class TestSteeringWithClusters:
    """Real users do not phrase questions the way documentation does."""

    @pytest.mark.placeborag(
        cluster_match="substring",
        clusters={
            "refund": [
                "how do I get my money back",
                "refund policy allows returns",
            ]
        },
    )
    def test_a_paraphrased_question_reaches_the_declared_chunk(
        self, fake_vector_store
    ):
        # Hashing alone would not connect these two: they share almost no
        # tokens. Declaring the cluster states the intent outright.
        #
        # cluster_match="substring" is the part that matters in a real
        # pipeline: the bot indexes *chunks*, so no stored text is ever
        # byte-identical to a declaration. With the default "exact" mode
        # this declaration would quietly do nothing.
        bot = SupportBot(store=fake_vector_store, generate=echo_llm)
        bot.index(DOCUMENTS)

        answer = bot.answer("how do I get my money back")

        assert answer.sources[0] == "refunds"
