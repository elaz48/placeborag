# Example: a support bot

A small RAG pipeline and the tests you can write against it with placeborag.

- `support_bot.py` — the code under test. It chunks documents, indexes them, retrieves, and generates an answer. It knows nothing about placeborag; the store arrives by injection, which is the only thing a pipeline has to do to become testable offline.
- `test_support_bot.py` — the tests. Every assertion here is one that passes vacuously against a random mock embedder.

```bash
pytest examples/
```

## What the tests demonstrate

**Retrieval you can assert on.** A refund question retrieves the refund document, and a shipping question does not. With random vectors both assertions are coin flips.

**Post-filtering that starves a filter.** Ask for Hungarian content with an English query and `filter_mode="post"`, and the pipeline returns *nothing* — the Hungarian chunks exist and match the filter, but they rank below the top-k cut, so filtering after the cut discards them. The same test under `pre` finds them.

**The reindex path.** An index built with one `model_name` and queried with another returns noise. This is the "we swapped the embedding model and forgot to reindex" incident, as a unit test.

**Chunking versus declarations.** `SupportBot` indexes *chunks*, so no stored text is ever byte-identical to a cluster declaration. That is what `cluster_match="substring"` is for — with the default `"exact"` mode the declaration would quietly do nothing. This example is the reason that mode exists.

## One test asserts a gap rather than a guarantee

`test_internal_content_never_leaks_into_a_public_answer` asserts that internal content *does* come back, because `SupportBot.answer` has no tier filter. The test documents the hole instead of hiding it, and its message says to fix the pipeline rather than the test. That is a real thing to do with a test: pin the behaviour you know is wrong, so nobody mistakes it for correct.
