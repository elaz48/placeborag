"""Failure injection on the retrieval path.

Everything here is deterministic. A fake that fails randomly would produce
flaky tests, which is the opposite of the point — you declare the failure,
and it happens the same way on every machine and every run.
"""

import pytest

from placeborag import (
    FakeEmbedder,
    FakeVectorStore,
    RetrievalError,
    RetrievalTimeout,
)

DOCUMENTS = [
    ("d1", "our refund policy allows returns within 30 days"),
    ("d2", "refund policy exceptions for sale items"),
    ("d3", "how to request a refund under the refund policy"),
    ("d4", "refund policy for digital purchases"),
    ("d5", "standard delivery times are three to five days"),
    ("d6", "express delivery arrives the next business day"),
]


def build(**kwargs):
    store = FakeVectorStore(embedder=FakeEmbedder(dimensions=64), **kwargs)
    for doc_id, text in DOCUMENTS:
        store.upsert(doc_id, text)
    return store


class TestDegradedRecall:
    """An ANN index does not promise the true top-k. Yours is tested as if
    it does."""

    def test_perfect_recall_is_the_default(self):
        assert len(build().query("refund policy", k=6)) == 6

    def test_degraded_recall_loses_candidates(self):
        degraded = build(recall=0.5).query("refund policy", k=6)

        assert len(degraded) < 6

    def test_zero_recall_returns_nothing(self):
        assert build(recall=0.0).query("refund policy", k=6) == []

    def test_the_same_query_loses_the_same_documents(self):
        store = build(recall=0.5)

        first = [match.id for match in store.query("refund policy", k=6)]
        second = [match.id for match in store.query("refund policy", k=6)]

        assert first == second

    def test_two_stores_with_the_same_recall_agree(self):
        left = [m.id for m in build(recall=0.5).query("refund policy", k=6)]
        right = [m.id for m in build(recall=0.5).query("refund policy", k=6)]

        assert left == right

    def test_different_queries_lose_different_documents(self):
        store = build(recall=0.5)

        refunds = [m.id for m in store.query("refund policy", k=6)]
        delivery = [m.id for m in store.query("delivery times", k=6)]

        assert refunds != delivery

    def test_survivors_keep_their_relative_order(self):
        perfect = [m.id for m in build().query("refund policy", k=6)]
        degraded = [m.id for m in build(recall=0.5).query("refund policy", k=6)]

        assert degraded == [doc_id for doc_id in perfect if doc_id in degraded]

    def test_recall_can_drop_the_top_result(self):
        # The point of the whole feature: the best document is not
        # guaranteed to come back, and a pipeline that assumes it does is
        # what this catches.
        best = build().query("refund policy", k=1)[0].id
        degraded = [m.id for m in build(recall=0.3).query("refund policy", k=6)]

        assert best not in degraded

    @pytest.mark.parametrize("recall", [-0.1, 1.1])
    def test_rejects_a_recall_outside_the_unit_interval(self, recall):
        with pytest.raises(ValueError, match="recall must be"):
            FakeVectorStore(recall=recall)


class TestStaleReads:
    """Writes are not always visible to the next read. Most pipelines assume
    they are."""

    def test_writes_are_visible_immediately_by_default(self):
        store = FakeVectorStore(embedder=FakeEmbedder(dimensions=64))
        store.upsert("d1", "refund policy")

        assert len(store.query("refund policy", k=1)) == 1

    def test_manual_visibility_hides_writes_until_refresh(self):
        store = FakeVectorStore(
            embedder=FakeEmbedder(dimensions=64), visibility="manual"
        )
        store.upsert("d1", "refund policy")

        assert store.query("refund policy", k=1) == []
        assert len(store) == 0
        assert store.pending_writes == 1

        store.refresh()

        assert len(store.query("refund policy", k=1)) == 1
        assert len(store) == 1
        assert store.pending_writes == 0

    def test_deletes_are_also_deferred(self):
        store = build(visibility="manual")
        store.refresh()

        store.delete("d1")

        assert "d1" in store.ids
        store.refresh()
        assert "d1" not in store.ids

    def test_refresh_is_idempotent(self):
        store = build(visibility="manual")
        store.refresh()
        store.refresh()

        assert len(store) == len(DOCUMENTS)

    def test_a_delete_of_an_invisible_write_still_resolves(self):
        store = FakeVectorStore(
            embedder=FakeEmbedder(dimensions=64), visibility="manual"
        )
        store.upsert("d1", "refund policy")
        store.delete("d1")
        store.refresh()

        assert store.ids == []

    def test_rejects_an_unknown_visibility(self):
        with pytest.raises(ValueError, match="visibility must be"):
            FakeVectorStore(visibility="eventual-ish")


class TestQueryFailures:
    def test_a_queued_failure_is_raised(self):
        store = build()
        store.fail_next_query(RetrievalTimeout("upstream timed out"))

        with pytest.raises(RetrievalTimeout, match="upstream timed out"):
            store.query("refund policy")

    def test_the_failure_applies_once_by_default(self):
        store = build()
        store.fail_next_query(RetrievalTimeout("once"))

        with pytest.raises(RetrievalTimeout):
            store.query("refund policy")

        assert len(store.query("refund policy", k=6)) == 6

    def test_a_failure_can_repeat(self):
        store = build()
        store.fail_next_query(RetrievalTimeout("twice"), times=2)

        for _ in range(2):
            with pytest.raises(RetrievalTimeout):
                store.query("refund policy")

        assert store.query("refund policy", k=1)

    def test_the_default_failure_is_a_timeout(self):
        store = build()
        store.fail_next_query()

        with pytest.raises(RetrievalTimeout):
            store.query("refund policy")

    def test_a_custom_exception_can_be_raised(self):
        class UpstreamRefused(Exception):
            pass

        store = build()
        store.fail_next_query(UpstreamRefused("connection refused"))

        with pytest.raises(UpstreamRefused):
            store.query("refund policy")

    def test_a_retrieval_timeout_is_a_retrieval_error(self):
        assert issubclass(RetrievalTimeout, RetrievalError)

    def test_writes_are_unaffected_by_a_queued_query_failure(self):
        store = build()
        store.fail_next_query()

        store.upsert("d7", "another refund policy document")

        assert "d7" in store.ids

    def test_rejects_a_non_positive_times(self):
        with pytest.raises(ValueError, match="times must be >= 1"):
            build().fail_next_query(times=0)


class TestFailuresCombine:
    def test_recall_and_stale_reads_stack(self):
        store = build(recall=0.5, visibility="manual")

        assert store.query("refund policy", k=6) == []

        store.refresh()
        matches = store.query("refund policy", k=6)

        assert 0 < len(matches) < 6

    def test_the_fixtures_accept_failure_options(self, fake_vector_store):
        # Marker options are derived from the constructor, so failure
        # settings work through the plugin without extra wiring.
        assert fake_vector_store.recall == 1.0
        assert fake_vector_store.visibility == "immediate"


@pytest.mark.placeborag(recall=0.5)
def test_recall_reaches_the_fixture(fake_vector_store):
    assert fake_vector_store.recall == 0.5
