# Your RAG tests are asserting that the pipeline ran

Not that it retrieved the right thing. There is a difference, and mock embedding tooling hides it.

## The setup everyone has

You are testing a RAG pipeline. You do not want to call OpenAI in CI, so you stub the embedding call. Every mock LLM library offers this, and they all do the same thing: return a vector of the right shape, filled with pseudo-random floats.

```python
def fake_embed(text: str) -> list[float]:
    return [random.random() for _ in range(1536)]
```

Correct dimensions. Correct type. Your pipeline runs end to end without a network call. The test passes.

Now look at what you can actually assert.

## What a random vector destroys

An embedding is only useful because *distance means something*. "refund policy" and "money back" should land near each other; "refund policy" and "delivery times to remote islands" should not. That geometry is the entire product. It is what makes top-k retrieval return relevant documents instead of arbitrary ones.

Random vectors have no geometry. Two random unit vectors in high dimensions are, with overwhelming probability, nearly orthogonal — regardless of what the underlying text says. Every document is equidistant from every query. Your ranking is a coin flip that lands differently on each run, or identically-but-meaninglessly if you seeded it.

So this assertion:

```python
results = retriever.search("refund policy", k=3)
assert results[0].id == "refund-doc"
```

is not testing retrieval. It is testing which arbitrary vector your RNG produced. If it passes, it passes for no reason. If someone breaks your chunking, your metadata filter, or your reranker, it keeps passing for the same no reason.

What you are left with is:

```python
assert len(results) == 3        # the pipeline ran
assert results[0].id is not None # ...and returned objects
```

That is a smoke test. Useful, but do not mistake it for a retrieval test.

## The bugs this hides

Three failures ship to production regularly, and none of them are catchable with random vectors, because all three are about *ordering and counts* — exactly what random vectors randomize.

### 1. The reversed sort

Chroma-style backends return a **distance**: lower is better. Qdrant-style backends return a **score**: higher is better. pgvector depends on which operator you chose.

Sort the wrong way and you retrieve the *least* relevant documents. The pipeline runs. The types are right. You get k results back. Every smoke test passes, and the answers are quietly garbage.

With random vectors this is undetectable — reversing a random order produces another random order.

### 2. Post-filtering that silently under-returns

You ask for `k=5` documents matching `{"lang": "hu"}`. There are six documents, two of which are Hungarian.

- **Pre-filtering** applies the filter, then takes the top 5. You get both Hungarian documents.
- **Post-filtering** takes the top 5, then applies the filter. If one Hungarian document ranked sixth, you get **one**.

Not an error. Not an empty result. A quietly incomplete answer, from a query that looks like it worked. Whether your backend pre-filters or post-filters is a configuration detail most teams have never checked.

You cannot write a unit test for this today. Reproducing it requires an embedder where you control which documents rank inside the top-k and which fall outside — precisely the geometry random vectors do not have.

### 3. Tie-breaking that differs between backends

Two documents with identical scores. Which comes first? Real stores disagree — insertion order, internal id, segment layout. Your test asserts on one order, CI runs against a different backend version, and the test flakes with no code change to blame.

## What to do instead

Stop making the embedder random. Make it *deterministic and steerable*.

A deterministic embedder is a pure function of its input: same text, same vector, every run, every machine. Build it with feature hashing — character n-grams and word tokens hashed into buckets, summed, normalized — and texts sharing tokens land near each other. No model, no data file, microseconds per call. The ranking is now meaningful *and explainable to whoever reads the failing test*, which is the part a real model does not give you either.

Then add explicit control for the cases where hashing is not enough:

```python
embedder = FakeEmbedder(clusters={
    "refund": ["refund policy", "money back", "hogyan kérek vissza pénzt"],
    "shipping": ["delivery times", "szállítási idő"],
})
```

Now the test *states its intent*. These three phrasings mean the same thing, so retrieval must treat them that way. If the geometry cannot satisfy the declaration, you find out when the embedder is constructed — not from a mysterious failure three assertions later.

And pair it with a store that reproduces backend quirks rather than an idealized one:

```python
pre = build("pre").query("refund policy", k=5, where={"lang": "hu"})
post = build("post").query("refund policy", k=5, where={"lang": "hu"})

assert len(pre) == 2
assert len(post) == 1  # the bug, now under test
```

A clean, ideal vector store catches nothing. The quirks *are* the product.

## The honest limitation

A deterministic fake embedder does not have real semantics. "car" and "automobile" share no characters, so hashing puts them nowhere near each other, while a real model knows they are synonyms. That is what declared clusters are for — you state the synonymy the test depends on.

This means a fake embedder cannot tell you whether your *retrieval quality* is good. Nothing offline can; that is what evals are for, and RAGAS and DeepEval do it well. What it tells you is whether your *retrieval plumbing* is correct: the sort direction, the filter ordering, the reindex path, the tie-breaking. Those are the parts that break silently, ship to production, and are currently untested everywhere.

Different question, different tool. Both worth answering.

---

This is the reasoning behind [placeborag](https://github.com/elaz48/placeborag), a small library of deterministic test doubles for the retrieval half of a RAG pipeline. `pip install placeborag`.
