"""Retrieval-quality metrics.

All functions follow the same contract:

- ``retrieved_ids``: document ids returned by the retriever, **in rank order**
  (best match first).
- ``relevant_ids``: the set of ids a human annotated as correct for the query.
- ``k``: how many of the top results we look at.

They are pure functions over ids — no I/O, no model calls — so they are
trivial to unit-test and they behave identically in CI and in production.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _validate(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int | None = None) -> None:
    if not relevant_ids:
        raise ValueError("relevant_ids is empty: every query in the QA set must be annotated")
    if k is not None and k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the relevant documents that show up in the top-k.

    "Of everything I should have found, how much did I find?"
    recall@k = |top-k ∩ relevant| / |relevant|
    """
    _validate(retrieved_ids, relevant_ids, k)
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results that are actually relevant.

    "Of everything I returned, how much was worth returning?"
    precision@k = |top-k ∩ relevant| / k

    Note the denominator is k (the budget), not the number of results:
    returning fewer documents than k does not buy a better precision.
    """
    _validate(retrieved_ids, relevant_ids, k)
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / k


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if at least one relevant document is in the top-k, else 0.0.

    The bluntest metric: "did the retriever give the generator at least
    one usable document?" Averaged over a QA set it reads as the share of
    queries that had any chance of being answered correctly.
    """
    _validate(retrieved_ids, relevant_ids, k)
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant document (0.0 if none found).

    Rewards putting a relevant document early: rank 1 → 1.0, rank 2 → 0.5,
    rank 3 → 0.33... Averaged over a QA set this is MRR (Mean Reciprocal
    Rank). Useful when the generator mostly reads the first documents.
    """
    _validate(retrieved_ids, relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance.

    Like recall, but position-aware: a relevant document at rank 1 is worth
    more than the same document at rank 5. Each hit at rank ``r`` contributes
    1/log2(r+1) (the "discount"), and the total is normalized by the best
    possible ordering (all relevant docs first), so the result is in [0, 1].
    """
    _validate(retrieved_ids, relevant_ids, k)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
