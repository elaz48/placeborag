"""Metadata filtering, compiled once per query.

A `where` clause is turned into a predicate before any record is examined, so
a malformed clause raises immediately instead of quietly matching nothing —
a filter that silently excludes everything is indistinguishable from one that
works, right up until production.

Two rules hold everywhere:

- A key the record does not carry never matches, whatever the operator.
  `{"lang": {"$ne": "hu"}}` will not surface records with no language at all.
- Comparing incompatible types raises. A numeric bound against a string is a
  bug in the filter, not a record that happens not to match.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping, Sequence

Predicate = Callable[[Mapping[str, object]], bool]

_MISSING = object()

_AND = "$and"
_OR = "$or"
_LOGICAL_OPERATORS = (_AND, _OR)

_COMPARISONS = {
    "$gt": operator.gt,
    "$gte": operator.ge,
    "$lt": operator.lt,
    "$lte": operator.le,
}
_MEMBERSHIP = ("$in", "$nin")
_EQUALITY = ("$eq", "$ne")
_FIELD_OPERATORS = (*_EQUALITY, *_MEMBERSHIP, *_COMPARISONS)


def compile_filter(where: Mapping[str, object] | None) -> Predicate:
    """Compiles a `where` clause into a predicate over a metadata mapping.

    Raises TypeError or ValueError on a malformed clause.
    """
    if where is None:
        return lambda metadata: True
    if not isinstance(where, Mapping):
        raise TypeError(f"where must be a mapping, got {type(where).__name__}")

    predicates = [_compile_entry(key, condition) for key, condition in where.items()]
    if not predicates:
        return lambda metadata: True
    return lambda metadata: all(predicate(metadata) for predicate in predicates)


def _compile_entry(key: str, condition: object) -> Predicate:
    if isinstance(key, str) and key.startswith("$"):
        return _compile_logical(key, condition)
    return _compile_field(key, condition)


def _compile_logical(name: str, clauses: object) -> Predicate:
    if name not in _LOGICAL_OPERATORS:
        raise ValueError(
            f"unknown operator {name!r}, expected one of "
            f"{', '.join(_LOGICAL_OPERATORS)}"
        )
    if isinstance(clauses, (str, bytes)) or not isinstance(clauses, Sequence):
        raise TypeError(f"{name} requires a list of clauses")
    if not clauses:
        raise ValueError(f"{name} requires at least one clause")

    compiled = [compile_filter(clause) for clause in clauses]
    combine = all if name == _AND else any
    return lambda metadata: combine(predicate(metadata) for predicate in compiled)


def _compile_field(key: str, condition: object) -> Predicate:
    if not _is_operator_mapping(condition):
        return lambda metadata: metadata.get(key, _MISSING) == condition

    checks = [
        _compile_operator(key, name, argument)
        for name, argument in condition.items()  # type: ignore[union-attr]
    ]
    return lambda metadata: all(check(metadata) for check in checks)


def _is_operator_mapping(condition: object) -> bool:
    """True for `{"$gt": 1}`, false for a plain nested value like `{"a": 1}`."""
    if not isinstance(condition, Mapping) or not condition:
        return False

    flagged = [str(key).startswith("$") for key in condition]
    if all(flagged):
        return True
    if any(flagged):
        raise ValueError(
            f"condition {condition!r} mixes operators with plain keys; "
            "use one or the other"
        )
    return False


def _compile_operator(key: str, name: str, argument: object) -> Predicate:
    if name in _MEMBERSHIP:
        if isinstance(argument, (str, bytes)) or not isinstance(argument, Sequence):
            raise TypeError(f"{name} requires a list of values, got {argument!r}")
        wanted = list(argument)
        negate = name == "$nin"
        return lambda metadata: _present_and(
            metadata, key, lambda value: (value in wanted) != negate
        )

    if name in _EQUALITY:
        negate = name == "$ne"
        return lambda metadata: _present_and(
            metadata, key, lambda value: (value == argument) != negate
        )

    compare = _COMPARISONS.get(name)
    if compare is None:
        raise ValueError(
            f"unknown operator {name!r}, expected one of "
            f"{', '.join(sorted(_FIELD_OPERATORS))}"
        )
    return lambda metadata: _present_and(
        metadata, key, lambda value: _compare(compare, key, value, argument)
    )


def _present_and(
    metadata: Mapping[str, object], key: str, check: Callable[[object], bool]
) -> bool:
    value = metadata.get(key, _MISSING)
    return value is not _MISSING and check(value)


def _compare(
    compare: Callable[[object, object], bool], key: str, value: object, bound: object
) -> bool:
    if _is_orderable_number(value) != _is_orderable_number(bound):
        raise TypeError(
            f"cannot compare {key}={value!r} with {bound!r}: incompatible types"
        )
    try:
        return compare(value, bound)
    except TypeError as error:
        raise TypeError(
            f"cannot compare {key}={value!r} with {bound!r}: {error}"
        ) from None


def _is_orderable_number(value: object) -> bool:
    # bool is an int in Python, but ordering a flag against a number is
    # almost always a mistake rather than an intention.
    return isinstance(value, (int, float)) and not isinstance(value, bool)
