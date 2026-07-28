# placeborag

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

## How it works

Character n-grams and word tokens are hashed into `dimensions` buckets with a signed hashing trick, summed, and normalized. Texts sharing tokens land near each other. There is no training data and no model file — the seed is derived from `(text, model_name, dimensions)`.

That makes the ranking explainable to whoever reads the failing test, which is the part random vectors can never give you.

## Status

0.0.1 is deliberately small: the hashing layer of `FakeEmbedder` and nothing else. It exists to prove the release pipeline.

Next up:

- declared clusters, so you can steer which texts are near which
- a fake vector store reproducing real backend score conventions (distance-lower-is-better vs score-higher-is-better) and pre-filter vs post-filter semantics
- pytest fixtures as a thin layer over the library

## What this is not

Not an eval framework, not a benchmark, not a production vector store, and not another OpenAI-compatible mock server.

## License

MIT
