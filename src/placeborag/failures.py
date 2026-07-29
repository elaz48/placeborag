"""Failure modes on the retrieval path.

Three things go wrong in production retrieval that a clean fake never
reproduces:

- **Degraded recall.** An ANN index does not promise the true top-k. It can
  miss the best document and return a worse one in its place, and nothing
  about the response says so.
- **Stale reads.** A write is not always visible to the next read. Pipelines
  that index and immediately query are built on an assumption their backend
  may not hold.
- **Query failures.** Timeouts and refusals, where the question is whether
  the pipeline degrades or collapses.

All of it is deterministic. A fake that fails randomly produces flaky tests,
which is the opposite of the point: you declare the failure, and it happens
identically on every run and every machine.
"""

from __future__ import annotations

import zlib

VISIBILITY_IMMEDIATE = "immediate"
VISIBILITY_MANUAL = "manual"
VISIBILITY_MODES = (VISIBILITY_IMMEDIATE, VISIBILITY_MANUAL)

PERFECT_RECALL = 1.0

_UINT32_RANGE = float(1 << 32)


class RetrievalError(Exception):
    """Base class for injected retrieval failures."""


class RetrievalTimeout(RetrievalError):
    """The retrieval call did not complete in time."""


def validate_recall(recall: float) -> float:
    if isinstance(recall, bool) or not isinstance(recall, (int, float)):
        raise TypeError("recall must be a number")
    if not 0.0 <= recall <= 1.0:
        raise ValueError(f"recall must be between 0.0 and 1.0, got {recall}")
    return float(recall)


def validate_visibility(visibility: str) -> str:
    if visibility not in VISIBILITY_MODES:
        raise ValueError(
            f"visibility must be one of {VISIBILITY_MODES}, got {visibility!r}"
        )
    return visibility


class RecallSampler:
    """Decides, deterministically, which candidates an index fails to return.

    The decision depends on the query and the record id, not on position or
    call order — so the same query loses the same documents every time, while
    a different query loses a different set. That is what an approximate
    index actually feels like.
    """

    def __init__(self, recall: float, seed: int) -> None:
        self._recall = recall
        self._seed = seed

    @property
    def is_perfect(self) -> bool:
        return self._recall >= PERFECT_RECALL

    def survives(self, query: str, record_id: str) -> bool:
        if self.is_perfect:
            return True
        if self._recall <= 0.0:
            return False
        digest = zlib.crc32(f"{query}\x00{record_id}".encode(), self._seed)
        return digest / _UINT32_RANGE < self._recall


class FailureQueue:
    """Exceptions queued to be raised by the next queries."""

    def __init__(self) -> None:
        self._pending: list[BaseException] = []

    def push(self, error: BaseException, times: int) -> None:
        if not isinstance(times, int) or isinstance(times, bool):
            raise TypeError("times must be an int")
        if times < 1:
            raise ValueError(f"times must be >= 1, got {times}")
        self._pending.extend([error] * times)

    def take(self) -> BaseException | None:
        return self._pending.pop(0) if self._pending else None

    def __len__(self) -> int:
        return len(self._pending)
