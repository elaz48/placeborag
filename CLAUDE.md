# placeborag

Deterministic test doubles for the retrieval half of a RAG pipeline: a steerable fake embedder and a fake vector store that reproduce real backend score conventions and filter semantics. See `PLAN.md` (gitignored, local only) for the roadmap and decision log.

Status: `0.4.0` on PyPI. M0–M3 complete.

## Commands

- Test: `pytest` (runs `tests/` and `examples/`)
- Lint: `ruff check src tests examples`
- Format: `ruff format src tests examples`
- Build: `python -m build`
- Install for development: `pip install -e ".[dev]"`

## Layout

- `src/placeborag/embedder.py` — `FakeEmbedder`, feature hashing via `crc32`
- `src/placeborag/clusters.py` — declared clusters, the control layer over hashing
- `src/placeborag/vector_store.py` — `FakeVectorStore`, backend profiles, filter modes
- `src/placeborag/filters.py` — `where` clause compilation
- `src/placeborag/failures.py` — degraded recall, stale reads, injected query errors
- `src/placeborag/pytest_plugin.py` — fixtures, registered via a `pytest11` entry point
- `examples/` — a worked RAG pipeline and its tests; runs in CI

## Invariants

- **Vectors are stable within a version, not across versions.** Assert on relative ordering, never on stored coordinates. Changing the hashing scheme is allowed pre-1.0 and has happened once.
- `FakeEmbedder` determinism is seeded from `(text, model_name, dimensions)`. Changing `model_name` must change the embedding space — that is what makes the reindex path testable.
- Quantities that must not vary with vector length use `_model_key` (model name only), not `_key` (model name plus dimensions). This is what keeps `cos(anchor, member) = 1/sqrt(1+m²)` free of a dimension term.
- Cluster declarations are verified geometrically at construction and raise when unsatisfiable. Failing loudly there beats a mysteriously failing retrieval assertion later.
- Malformed filters raise; they never quietly match nothing. A filter that silently excludes everything is indistinguishable from one that works.
- An absent metadata key never matches, under any operator.
- **Injected failures are deterministic, never random.** A randomly failing fake produces flaky tests, which is the failure mode this library exists to eliminate. Recall survival is keyed on `(query, record_id)`.

## Releasing

The tag is the publish trigger, and PyPI will not accept a re-upload of a version. In order:

1. Update the README "Status" section — the README at tag time is what PyPI displays for that version, forever.
2. Bump `__version__` in `src/placeborag/__init__.py`, the only place it lives.
3. Run the suite the way CI does: `pip install -e ".[dev]"` in a clean venv.
4. Push `main`, wait for CI to go green.
5. Tag `vX.Y.Z` and push it. Publishing happens via Trusted Publishing (OIDC), no API tokens.
6. Create the GitHub release.

Skipping step 1 is how `0.0.2` shipped with a README claiming its own features were unreleased.

## Watch for

Duplicated sources of truth have caused three separate failures here: the version in two files, the CI test-dependency list, and the plugin's marker-option list. All three now derive from one place. Prefer deriving over listing.
