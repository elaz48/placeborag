"""The control layer: declared clusters.

A retrieval test should be a statement of intent, not a lottery. Declaring a
cluster is that statement, so the guarantees here are strict and violations
are loud.
"""

import itertools
import math

import pytest

from placeborag import FakeEmbedder, cosine_similarity

CLUSTERS = {
    "refund": ["refund policy", "money back", "hogyan kérek vissza pénzt"],
    "shipping": ["delivery times", "szállítási idő"],
}


def _members(clusters):
    return [(name, text) for name, texts in clusters.items() for text in texts]


class TestClusterGuarantee:
    def test_members_outrank_non_members_for_every_pair(self):
        embedder = FakeEmbedder(clusters=CLUSTERS)
        vectors = {text: embedder.embed(text) for _, text in _members(CLUSTERS)}

        intra = [
            cosine_similarity(vectors[a], vectors[b])
            for texts in CLUSTERS.values()
            for a, b in itertools.combinations(texts, 2)
        ]
        inter = [
            cosine_similarity(vectors[a], vectors[b])
            for (left, a), (right, b) in itertools.combinations(_members(CLUSTERS), 2)
            if left != right
        ]

        assert min(intra) > max(inter)

    def test_a_declared_member_beats_an_undeclared_lookalike(self):
        embedder = FakeEmbedder(clusters=CLUSTERS)

        query = embedder.embed("refund policy")
        sibling = embedder.embed("money back")
        outsider = embedder.embed("something entirely unrelated to any cluster")

        assert cosine_similarity(query, sibling) > cosine_similarity(query, outsider)

    def test_undeclared_text_falls_through_to_the_hashing_layer(self):
        plain = FakeEmbedder()
        clustered = FakeEmbedder(clusters=CLUSTERS)

        assert clustered.embed("undeclared text") == plain.embed("undeclared text")

    def test_declared_text_does_not_use_the_hashing_layer(self):
        plain = FakeEmbedder()
        clustered = FakeEmbedder(clusters=CLUSTERS)

        assert clustered.embed("refund policy") != plain.embed("refund policy")


class TestClusterDeterminism:
    def test_the_same_declaration_gives_the_same_vectors(self):
        left = FakeEmbedder(clusters=CLUSTERS)
        right = FakeEmbedder(clusters=CLUSTERS)

        assert left.embed("refund policy") == right.embed("refund policy")

    def test_cluster_vectors_are_normalized(self):
        embedder = FakeEmbedder(clusters=CLUSTERS)

        for _, text in _members(CLUSTERS):
            vector = embedder.embed(text)
            assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)

    def test_model_name_still_changes_the_space(self):
        left = FakeEmbedder(model_name="a", clusters=CLUSTERS)
        right = FakeEmbedder(model_name="b", clusters=CLUSTERS)

        assert left.embed("refund policy") != right.embed("refund policy")

    def test_dimensions_do_not_change_ordering_within_a_cluster(self):
        # Ranking members against their anchor is exactly 1/sqrt(1+m**2),
        # which has no dimension term — so the order is a property of the
        # declaration, not of the vector length it happens to run at.
        orders = []
        for dimensions in (64, 256, 1024):
            embedder = FakeEmbedder(dimensions=dimensions, clusters=CLUSTERS)
            anchor = embedder.cluster_anchor("refund")
            ranked = sorted(
                CLUSTERS["refund"],
                key=lambda text: -cosine_similarity(anchor, embedder.embed(text)),
            )
            orders.append(ranked)

        assert orders[0] == orders[1] == orders[2]

    def test_member_similarity_to_its_anchor_is_dimension_independent(self):
        similarities = []
        for dimensions in (64, 256, 1024):
            embedder = FakeEmbedder(dimensions=dimensions, clusters=CLUSTERS)
            similarities.append(
                cosine_similarity(
                    embedder.cluster_anchor("refund"), embedder.embed("money back")
                )
            )

        assert similarities[0] == pytest.approx(similarities[1], abs=1e-12)
        assert similarities[1] == pytest.approx(similarities[2], abs=1e-12)

    def test_anchor_is_unknown_for_an_undeclared_cluster(self):
        assert FakeEmbedder(clusters=CLUSTERS).cluster_anchor("nope") is None
        assert FakeEmbedder().cluster_anchor("refund") is None

    def test_cluster_membership_is_reported(self):
        embedder = FakeEmbedder(clusters=CLUSTERS)

        assert embedder.cluster_of("money back") == "refund"
        assert embedder.cluster_of("undeclared") is None
        assert embedder.cluster_names == ["refund", "shipping"]


class TestSubstringMatching:
    """Real pipelines chunk their documents, so the indexed text is never
    byte-identical to what a test declared."""

    CHUNK = "Our refund policy allows returns within 30 days of purchase. Refunds are"

    def _embedder(self, match):
        return FakeEmbedder(
            clusters={"refund": ["refund policy"], "shipping": ["delivery times"]},
            cluster_match=match,
        )

    def test_exact_matching_ignores_a_chunk_that_merely_contains_the_member(self):
        embedder = self._embedder("exact")
        plain = FakeEmbedder()

        # Falls through to hashing, which is correct for exact mode but is
        # the trap that makes a declaration look like it did nothing.
        assert embedder.embed(self.CHUNK) == plain.embed(self.CHUNK)

    def test_substring_matching_pulls_the_chunk_into_the_cluster(self):
        embedder = self._embedder("substring")

        anchor = embedder.cluster_anchor("refund")
        other = embedder.cluster_anchor("shipping")

        assert cosine_similarity(anchor, embedder.embed(self.CHUNK)) > cosine_similarity(
            other, embedder.embed(self.CHUNK)
        )

    def test_a_matched_chunk_still_gets_its_own_vector(self):
        embedder = self._embedder("substring")

        assert embedder.embed(self.CHUNK) != embedder.embed("refund policy")
        assert embedder.embed(self.CHUNK) != embedder.embed(self.CHUNK + " extra")

    def test_matching_is_deterministic(self):
        left = self._embedder("substring")
        right = self._embedder("substring")

        assert left.embed(self.CHUNK) == right.embed(self.CHUNK)

    def test_unrelated_text_still_falls_through_to_hashing(self):
        embedder = self._embedder("substring")

        assert embedder.embed("nothing to do with any of it") == FakeEmbedder().embed(
            "nothing to do with any of it"
        )

    def test_the_most_specific_declaration_wins(self):
        embedder = FakeEmbedder(
            clusters={
                "general": ["refund"],
                "specific": ["refund policy for digital purchases"],
            },
            cluster_match="substring",
        )
        chunk = "see the refund policy for digital purchases below"

        specific = embedder.cluster_anchor("specific")
        general = embedder.cluster_anchor("general")

        assert cosine_similarity(specific, embedder.embed(chunk)) > cosine_similarity(
            general, embedder.embed(chunk)
        )

    def test_cluster_of_reports_a_substring_match(self):
        embedder = self._embedder("substring")

        assert embedder.cluster_of(self.CHUNK) == "refund"
        assert embedder.cluster_of("unrelated") is None

    def test_a_matched_chunk_is_nearer_its_cluster_than_another_cluster(self):
        embedder = self._embedder("substring")

        chunk_vector = embedder.embed(self.CHUNK)

        assert cosine_similarity(
            chunk_vector, embedder.embed("refund policy")
        ) > cosine_similarity(chunk_vector, embedder.embed("delivery times"))

    def test_the_anchor_cosine_stays_dimension_independent(self):
        # The offset is now the hashing vector rather than a generated
        # direction, but it is still orthogonalized against the anchor — so
        # cos(anchor, derived) is 1/sqrt(1+m**2) with no dimension term.
        similarities = []
        for dimensions in (64, 256, 1024):
            embedder = FakeEmbedder(
                dimensions=dimensions,
                clusters={"refund": ["refund policy"]},
                cluster_match="substring",
            )
            similarities.append(
                cosine_similarity(
                    embedder.cluster_anchor("refund"), embedder.embed(self.CHUNK)
                )
            )

        assert similarities[0] == pytest.approx(similarities[1], abs=1e-12)
        assert similarities[1] == pytest.approx(similarities[2], abs=1e-12)

    def test_chunks_that_hash_alike_stay_alike_inside_the_cluster(self):
        # Using the hashing vector as the offset means related chunks keep
        # their relationship inside the cluster, instead of being scattered.
        embedder = self._embedder("substring")

        near = embedder.embed("refund policy applies to returns within 30 days")
        also_near = embedder.embed("refund policy applies to returns within 60 days")
        further = embedder.embed("refund policy and nothing else whatsoever here")

        assert cosine_similarity(near, also_near) > cosine_similarity(near, further)

    def test_rejects_an_unknown_match_mode(self):
        with pytest.raises(ValueError, match="cluster_match must be"):
            FakeEmbedder(clusters=CLUSTERS, cluster_match="fuzzy")


class TestLoudValidation:
    def test_rejects_a_text_declared_in_two_clusters(self):
        with pytest.raises(ValueError, match="declared in more than one cluster"):
            FakeEmbedder(
                clusters={"a": ["shared text"], "b": ["shared text", "other"]}
            )

    def test_rejects_an_empty_cluster(self):
        with pytest.raises(ValueError, match="at least one member"):
            FakeEmbedder(clusters={"refund": []})

    def test_rejects_an_empty_cluster_name(self):
        with pytest.raises(ValueError, match="cluster name"):
            FakeEmbedder(clusters={"": ["text"]})

    def test_rejects_non_string_members(self):
        with pytest.raises(TypeError, match="cluster member"):
            FakeEmbedder(clusters={"refund": ["ok", 42]})

    def test_rejects_a_non_mapping_clusters_argument(self):
        with pytest.raises(TypeError, match="clusters must be"):
            FakeEmbedder(clusters=["refund", "shipping"])

    @pytest.mark.parametrize("spread", [0.0, -0.1, 1.5])
    def test_rejects_an_out_of_range_spread(self, spread):
        with pytest.raises(ValueError, match="cluster_spread"):
            FakeEmbedder(clusters=CLUSTERS, cluster_spread=spread)

    def test_a_string_instead_of_a_member_list_is_rejected(self):
        # {"refund": "refund policy"} would silently cluster each character.
        with pytest.raises(TypeError, match="cluster member"):
            FakeEmbedder(clusters={"refund": "refund policy"})
