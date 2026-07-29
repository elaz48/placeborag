"""Metadata filter semantics.

Two rules run through all of this:

- A key the record does not carry never matches, for any operator. "Absent"
  is not a value to compare against.
- A malformed clause raises rather than quietly matching nothing. A filter
  that silently excludes everything looks exactly like a filter that works.
"""

import pytest

from placeborag import FakeEmbedder, FakeVectorStore
from placeborag.filters import compile_filter

RECORDS = [
    ("a", {"lang": "en", "year": 2024, "tier": "public", "score": 4.5}),
    ("b", {"lang": "hu", "year": 2025, "tier": "public", "score": 2.0}),
    ("c", {"lang": "de", "year": 2023, "tier": "internal", "score": 9.0}),
    ("d", {"lang": "en", "year": 2026, "tier": "internal"}),
]


def matching(where):
    predicate = compile_filter(where)
    return [name for name, metadata in RECORDS if predicate(metadata)]


class TestEquality:
    def test_a_bare_value_means_equality(self):
        assert matching({"lang": "en"}) == ["a", "d"]

    def test_several_keys_are_combined_with_and(self):
        assert matching({"lang": "en", "tier": "public"}) == ["a"]

    def test_an_explicit_eq_operator_matches_the_bare_form(self):
        assert matching({"lang": {"$eq": "en"}}) == matching({"lang": "en"})

    def test_a_missing_key_never_matches(self):
        assert matching({"score": 4.5}) == ["a"]
        assert matching({"absent": "anything"}) == []

    def test_a_nested_mapping_without_operators_is_a_literal_value(self):
        predicate = compile_filter({"meta": {"nested": 1}})

        assert predicate({"meta": {"nested": 1}})
        assert not predicate({"meta": {"nested": 2}})


class TestMembership:
    def test_in_matches_any_listed_value(self):
        assert matching({"lang": {"$in": ["hu", "de"]}}) == ["b", "c"]

    def test_nin_excludes_listed_values(self):
        assert matching({"lang": {"$nin": ["hu", "de"]}}) == ["a", "d"]

    def test_nin_still_requires_the_key_to_exist(self):
        # "d" has no score, so it is absent from both results — $nin is not
        # a way to select records missing the field.
        assert matching({"score": {"$nin": [4.5]}}) == ["b", "c"]

    def test_ne_excludes_the_value(self):
        assert matching({"tier": {"$ne": "internal"}}) == ["a", "b"]

    @pytest.mark.parametrize("operator", ["$in", "$nin"])
    def test_membership_operators_require_a_sequence(self, operator):
        with pytest.raises(TypeError, match="requires a list"):
            compile_filter({"lang": {operator: "hu"}})


class TestComparison:
    def test_gte_and_lte_bound_a_range(self):
        assert matching({"year": {"$gte": 2024, "$lte": 2025}}) == ["a", "b"]

    def test_gt_and_lt_are_exclusive(self):
        assert matching({"year": {"$gt": 2024, "$lt": 2026}}) == ["b"]

    def test_comparisons_work_on_floats(self):
        assert matching({"score": {"$gt": 3.0}}) == ["a", "c"]

    def test_comparing_incompatible_types_raises(self):
        # A string year against a numeric bound is a bug in the test that
        # wrote the filter, not a record that fails to match.
        predicate = compile_filter({"year": {"$gt": 2024}})

        with pytest.raises(TypeError, match="cannot compare"):
            predicate({"year": "recent"})

    def test_booleans_are_not_treated_as_numbers(self):
        with pytest.raises(TypeError, match="cannot compare"):
            compile_filter({"flag": {"$gt": 1}})({"flag": True})


class TestLogicalOperators:
    def test_or_matches_any_branch(self):
        assert matching({"$or": [{"lang": "hu"}, {"tier": "internal"}]}) == [
            "b",
            "c",
            "d",
        ]

    def test_and_matches_every_branch(self):
        assert matching({"$and": [{"lang": "en"}, {"year": {"$gt": 2025}}]}) == ["d"]

    def test_logical_operators_nest(self):
        where = {
            "$or": [
                {"$and": [{"lang": "en"}, {"tier": "public"}]},
                {"year": {"$lt": 2024}},
            ]
        }

        assert matching(where) == ["a", "c"]

    @pytest.mark.parametrize("operator", ["$and", "$or"])
    def test_logical_operators_require_a_non_empty_list(self, operator):
        with pytest.raises(TypeError, match="requires a list"):
            compile_filter({operator: {"lang": "hu"}})
        with pytest.raises(ValueError, match="at least one"):
            compile_filter({operator: []})


class TestMalformedClauses:
    def test_an_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="unknown operator"):
            compile_filter({"year": {"$approximately": 2024}})

    def test_an_unknown_top_level_operator_raises(self):
        with pytest.raises(ValueError, match="unknown operator"):
            compile_filter({"$nor": [{"lang": "hu"}]})

    def test_mixing_operators_and_plain_keys_raises(self):
        with pytest.raises(ValueError, match="mixes operators"):
            compile_filter({"year": {"$gt": 2024, "nested": 1}})

    def test_a_non_mapping_where_raises(self):
        with pytest.raises(TypeError, match="where must be a mapping"):
            compile_filter([("lang", "hu")])

    def test_an_empty_where_matches_everything(self):
        assert matching({}) == ["a", "b", "c", "d"]


class TestThroughTheStore:
    def _store(self, **kwargs):
        store = FakeVectorStore(embedder=FakeEmbedder(dimensions=32), **kwargs)
        for name, metadata in RECORDS:
            store.upsert(name, f"document {name} about refunds", metadata=metadata)
        return store

    def test_the_store_accepts_operator_filters(self):
        store = self._store()

        matches = store.query("refunds", k=4, where={"year": {"$gte": 2025}})

        assert sorted(match.id for match in matches) == ["b", "d"]

    def test_operator_filters_work_under_post_filtering(self):
        store = self._store(filter_mode="post")

        matches = store.query("refunds", k=1, where={"lang": {"$in": ["hu", "de"]}})

        # Post-filtering cuts to k=1 first, so at most one record survives
        # and it may be none.
        assert len(matches) <= 1

    def test_a_malformed_filter_raises_at_query_time(self):
        store = self._store()

        with pytest.raises(ValueError, match="unknown operator"):
            store.query("refunds", where={"year": {"$roughly": 2024}})
