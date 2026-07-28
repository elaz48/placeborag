# placeborag

[![ci](https://github.com/elaz48/placeborag/actions/workflows/ci.yml/badge.svg)](https://github.com/elaz48/placeborag/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/placeborag.svg)](https://pypi.org/project/placeborag/)
[![Python versions](https://img.shields.io/pypi/pyversions/placeborag.svg)](https://pypi.org/project/placeborag/)

Deterministic test doubles for the **retrieval** half of a RAG pipeline.

Existing mock LLM tooling stubs the chat completion endpoint and hands back pseudo-random embedding vectors of the correct shape. Correct shape, no semantic structure. Any document can come back at any rank, so every retrieval assertion in your test suite is decorative: you can assert that the pipeline ran, not that it retrieved the right thing.

placeborag gives you an embedder that is a pure function whose output geometry you can reason about — offline, in microseconds, with no model and no network.

## Install

```bash
pip install placeborag
```

## Use

```python
from placeborag import FakeEmbedder, cosine_similarity

embedder = FakeEmbedder()

query = embedder.embed("refund policy")
related = embedder.embed("what is your refund policy for orders")
unrelated = embedder.embed("delivery times to remote islands")

assert cosine_similarity(query, related) > cosine_similarity(query, unrelated)
```

The same text always gives the same vector, so a top-k assertion is stable across runs and across machines:

```python
assert embedder.embed("refund policy") == embedder.embed("refund policy")
```

Changing `model_name` changes the embedding space, which is what makes the "we swapped the embedding model and have to reindex" code path testable:

```python
a = FakeEmbedder(model_name="text-embedding-3-small")
b = FakeEmbedder(model_name="text-embedding-3-large")

assert a.embed("refund policy") != b.embed("refund policy")
```

Every vector is L2-normalized, including for the empty string and for non-ASCII input.

## Steering what is near what

Hashing gets you meaningful ordering for free, but sometimes a test needs to state outright that these three phrasings mean the same thing. Declare a cluster:

```python
from placeborag import FakeEmbedder, cosine_similarity

embedder = FakeEmbedder(clusters={
    "refund": ["refund policy", "money back", "hogyan kérek vissza pénzt"],
    "shipping": ["delivery times", "szállítási idő"],
})

assert cosine_similarity(
    embedder.embed("refund policy"), embedder.embed("money back")
) > cosine_similarity(
    embedder.embed("refund policy"), embedder.embed("delivery times")
)
```

Any text you did not declare falls through to the hashing layer unchanged, so clusters are additive — you steer the handful of phrases the test is actually about and leave the rest alone.

The declaration is checked when the embedder is constructed, not when a test later fails mysteriously. If the geometry cannot satisfy what you declared, you get a `ValueError` naming the offending similarities:

```python
FakeEmbedder(clusters={"a": ["same text"], "b": ["same text"]})
# ValueError: 'same text' is declared in more than one cluster ('a' and 'b')
```

Each cluster has an anchor you can query with. The cosine between an anchor and one of its members is exactly `1 / sqrt(1 + jitter**2)` — no dimension term — so ranking a cluster's members against their anchor gives the same order at `dimensions=64` and `dimensions=1024`:

```python
anchor = embedder.cluster_anchor("refund")
ranked = sorted(
    ["refund policy", "money back"],
    key=lambda text: -cosine_similarity(anchor, embedder.embed(text)),
)
```

Ranking members against *each other* does not carry that guarantee: that angle involves two jitter directions, and it does depend on the dimension.

## The bug you cannot currently unit test

Post-filtering applies your metadata filter *after* the top-k cut, so it can return fewer results than you asked for. Same query, same `k`, same data — different answer, depending only on which backend you are pointed at:

```python
from placeborag import FakeEmbedder, FakeVectorStore

KNOWLEDGE_BASE = [
    ("en-1", "our refund policy allows returns within 30 days", "en"),
    ("en-2", "refund policy exceptions for sale items", "en"),
    ("en-3", "how to request a refund under the refund policy", "en"),
    ("en-4", "refund policy for digital purchases", "en"),
    ("hu-1", "pénzvisszatérítési szabályzat harminc napon belül", "hu"),
    ("hu-2", "hogyan kérek vissza pénzt vásárlás után", "hu"),
]

def build(filter_mode):
    store = FakeVectorStore(embedder=FakeEmbedder(), filter_mode=filter_mode)
    for doc_id, text, lang in KNOWLEDGE_BASE:
        store.upsert(doc_id, text, metadata={"lang": lang})
    return store

pre = build("pre").query("refund policy", k=5, where={"lang": "hu"})
post = build("post").query("refund policy", k=5, where={"lang": "hu"})

assert len(pre) == 2   # both Hungarian documents
assert len(post) == 1  # one of them fell outside the top-5 before filtering
```

That second assertion is the bug that ships to production. It is not a crash and not an empty result — just a quietly incomplete answer.

### Score conventions

Chroma-style backends return a **distance**, where lower is better. Qdrant-style backends return a **score**, where higher is better. Point the same code at the other one and your sort is reversed:

```python
from placeborag import FakeVectorStore

chroma = FakeVectorStore(profile="chroma")  # distance, lower is better
qdrant = FakeVectorStore(profile="qdrant")  # similarity, higher is better

assert chroma.profile.higher_is_better is False
assert qdrant.profile.higher_is_better is True
```

Both profiles rank documents in the same relevance order — only the reported number differs. A test that passes on one profile and fails on the other has a sort direction bug.

Ties are broken deterministically, and the two profiles break them differently: Chroma-style by insertion order, Qdrant-style by id. Real stores differ here too, and the difference stays invisible until a test flakes in CI.

## pytest fixtures

Installing placeborag registers two fixtures. They are a thin layer over the library — everything below is reachable from a plain script or a notebook too.

```python
def test_retrieval(fake_vector_store):
    fake_vector_store.upsert("doc", "our refund policy allows returns")

    matches = fake_vector_store.query("refund policy", k=1)

    assert matches[0].id == "doc"
```

Configure them per test with the `placeborag` marker:

```python
@pytest.mark.placeborag(
    profile="qdrant",
    filter_mode="post",
    clusters={"refund": ["refund policy", "money back"]},
)
def test_post_filtering_under_returns(fake_vector_store):
    ...
```

Embedder options (`model_name`, `dimensions`, `clusters`, `cluster_spread`) and store options (`profile`, `filter_mode`) go in the same marker and are routed to the right object. A misspelled option raises instead of being silently ignored.

## How it works

Character n-grams and word tokens are hashed into `dimensions` buckets with a signed hashing trick, summed, and normalized. Texts sharing tokens land near each other. There is no training data and no model file — the seed is derived from `(text, model_name, dimensions)`.

That makes the ranking explainable to whoever reads the failing test, which is the part random vectors can never give you.

Embedding 10,000 short strings takes well under a second on one core, so a full test suite can embed freely without a fixture cache.

## Status

`0.1.0` is the current release and contains everything documented above.

Pre-1.0 in the way that matters: **vectors are stable within a version, not across versions.** Assert on relative ordering, never on stored coordinates. (`0.0.1` produces different coordinates than later versions; the ordering behaviour is the same.)

Next up:

- more backend profiles: Qdrant is modelled, FAISS and Weaviate are not
- richer metadata filters — `where` is equality-only today
- failure injection on the retrieval path: timeouts, partial index, degraded recall

## Why this exists

Longer version, with the three bugs random mock vectors hide: [Your RAG tests are asserting that the pipeline ran](docs/why-random-vectors-fail.md).

## What this is not

Not an eval framework, not a benchmark, not a production vector store, and not another OpenAI-compatible mock server.

A fake embedder cannot tell you whether your retrieval *quality* is good — nothing offline can, and RAGAS and DeepEval occupy that space properly. It tells you whether your retrieval *plumbing* is correct: sort direction, filter ordering, the reindex path, tie-breaking.

## License

MIT
