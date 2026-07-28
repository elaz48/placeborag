"""The bug this library exists for: post-filtering silently returns fewer
results than you asked for, and there is currently no way to unit test it."""

import pytest

from placeborag import FakeEmbedder, FakeVectorStore

# Four English documents that all match a "refund policy" query strongly, and
# two Hungarian ones that share no tokens with it. With k=5 exactly one
# document falls outside the top-k, and it is a Hungarian one.
KNOWLEDGE_BASE = [
    ("en-1", "our refund policy allows returns within 30 days", "en"),
    ("en-2", "refund policy exceptions for sale items", "en"),
    ("en-3", "how to request a refund under the refund policy", "en"),
    ("en-4", "refund policy for digital purchases", "en"),
    ("hu-1", "pénzvisszatérítési szabályzat harminc napon belül", "hu"),
    ("hu-2", "hogyan kérek vissza pénzt vásárlás után", "hu"),
]


def _store(filter_mode):
    return _populate(FakeVectorStore(embedder=FakeEmbedder(), filter_mode=filter_mode))


def _store_with_profile(profile):
    return _populate(FakeVectorStore(embedder=FakeEmbedder(), profile=profile))


def _populate(store):
    for doc_id, text, lang in KNOWLEDGE_BASE:
        store.upsert(doc_id, text, metadata={"lang": lang})
    return store


def test_pre_filter_returns_every_matching_document():
    matches = _store("pre").query("refund policy", k=5, where={"lang": "hu"})

    # Which Hungarian document ranks first is the embedder's business; what
    # this test asserts is that pre-filtering loses neither of them.
    assert {match.id for match in matches} == {"hu-1", "hu-2"}


def test_post_filter_silently_drops_matches_outside_the_top_k():
    matches = _store("post").query("refund policy", k=5, where={"lang": "hu"})

    # Same query, same k, same data — one result instead of two, because
    # filtering happened after the top-k cut. This ships to production.
    assert len(matches) == 1


def test_ranking_is_stable_across_calls():
    store = _store("pre")

    first = [match.id for match in store.query("refund policy", k=6)]
    second = [match.id for match in store.query("refund policy", k=6)]

    assert first == second


class TestScoreConventions:
    """Reversing the sort is the single most common retrieval bug."""

    def test_chroma_reports_a_distance_where_lower_is_better(self):
        store = _store_with_profile("chroma")

        matches = store.query("refund policy", k=6)

        assert store.profile.higher_is_better is False
        assert matches[0].score < matches[-1].score

    def test_qdrant_reports_a_score_where_higher_is_better(self):
        store = _store_with_profile("qdrant")

        matches = store.query("refund policy", k=6)

        assert store.profile.higher_is_better is True
        assert matches[0].score > matches[-1].score

    def test_the_two_profiles_agree_on_relevance_order(self):
        chroma = _store_with_profile("chroma").query("refund policy", k=6)
        qdrant = _store_with_profile("qdrant").query("refund policy", k=6)

        # Only the reported number differs. A test that passes on one profile
        # and fails on the other has a sort direction bug, not a data problem.
        assert [m.id for m in chroma] == [m.id for m in qdrant]

    def test_the_two_profiles_report_complementary_numbers(self):
        chroma = _store_with_profile("chroma").query("refund policy", k=1)[0]
        qdrant = _store_with_profile("qdrant").query("refund policy", k=1)[0]

        assert chroma.score == pytest.approx(1.0 - qdrant.score)


class TestTieBreaking:
    """Real stores differ here, and the difference is invisible until CI flakes."""

    def _tied_store(self, profile):
        store = FakeVectorStore(embedder=FakeEmbedder(), profile=profile)
        store.upsert("b", "identical text", metadata={})
        store.upsert("a", "identical text", metadata={})
        return store

    def test_chroma_breaks_ties_by_insertion_order(self):
        matches = self._tied_store("chroma").query("identical text", k=2)

        assert [match.id for match in matches] == ["b", "a"]

    def test_qdrant_breaks_ties_by_id(self):
        matches = self._tied_store("qdrant").query("identical text", k=2)

        assert [match.id for match in matches] == ["a", "b"]

    def test_ties_are_scored_identically_despite_the_order(self):
        matches = self._tied_store("chroma").query("identical text", k=2)

        assert matches[0].score == pytest.approx(matches[1].score)


class TestUpsertAndDelete:
    def test_upsert_replaces_the_record_under_the_same_id(self):
        store = FakeVectorStore()
        store.upsert("doc", "delivery times", metadata={"v": 1})
        store.upsert("doc", "refund policy", metadata={"v": 2})

        matches = store.query("refund policy", k=1)

        assert len(store) == 1
        assert matches[0].text == "refund policy"
        assert matches[0].metadata == {"v": 2}

    def test_upsert_keeps_the_original_insertion_position(self):
        store = FakeVectorStore(profile="chroma")
        store.upsert("b", "identical text")
        store.upsert("a", "identical text")
        store.upsert("b", "identical text", metadata={"touched": True})

        matches = store.query("identical text", k=2)

        # "b" was inserted first and stays first under an insertion-order
        # profile, even though it was written most recently.
        assert [match.id for match in matches] == ["b", "a"]

    def test_delete_removes_the_record(self):
        store = _store("pre")

        assert store.delete("hu-1") is True
        assert "hu-1" not in store.ids
        assert len(store) == len(KNOWLEDGE_BASE) - 1

    def test_delete_reports_a_missing_id(self):
        assert FakeVectorStore().delete("never-existed") is False

    def test_deleted_records_stop_matching(self):
        store = _store("pre")
        store.delete("hu-1")

        matches = store.query("refund policy", k=5, where={"lang": "hu"})

        assert [match.id for match in matches] == ["hu-2"]


class TestQuerySemantics:
    def test_empty_store_returns_no_matches(self):
        assert FakeVectorStore().query("refund policy") == []

    def test_k_caps_the_number_of_matches(self):
        assert len(_store("pre").query("refund policy", k=2)) == 2

    def test_a_where_key_missing_from_metadata_never_matches(self):
        matches = _store("pre").query("refund policy", k=6, where={"absent": "x"})

        assert matches == []

    def test_every_where_key_must_match(self):
        store = FakeVectorStore()
        store.upsert("doc", "refund policy", metadata={"lang": "hu", "tier": "free"})

        assert store.query("refund policy", where={"lang": "hu", "tier": "free"})
        assert store.query("refund policy", where={"lang": "hu", "tier": "paid"}) == []

    def test_match_metadata_is_a_copy(self):
        store = FakeVectorStore()
        store.upsert("doc", "refund policy", metadata={"lang": "hu"})

        store.query("refund policy")[0].metadata["lang"] = "mutated"

        assert store.query("refund policy")[0].metadata == {"lang": "hu"}


class TestValidation:
    def test_rejects_an_unknown_profile(self):
        with pytest.raises(ValueError, match="unknown profile"):
            FakeVectorStore(profile="pinecone")

    def test_rejects_an_unknown_filter_mode(self):
        with pytest.raises(ValueError, match="filter_mode must be"):
            FakeVectorStore(filter_mode="postfilter")

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_a_non_positive_k(self, k):
        with pytest.raises(ValueError, match="k must be >= 1"):
            FakeVectorStore().query("refund policy", k=k)

    def test_rejects_an_empty_id(self):
        with pytest.raises(ValueError, match="id must be"):
            FakeVectorStore().upsert("", "refund policy")

    def test_rejects_non_string_text(self):
        with pytest.raises(TypeError, match="text must be a str"):
            FakeVectorStore().upsert("doc", None)

    def test_repr_shows_the_configuration(self):
        store = FakeVectorStore(profile="qdrant", filter_mode="post")
        store.upsert("doc", "refund policy")

        assert repr(store) == (
            "FakeVectorStore(profile='qdrant', filter_mode='post', records=1)"
        )
