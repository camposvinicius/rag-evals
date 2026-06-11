"""Run an evaluation: retrieve for every QA example, score, aggregate.

The output is a plain dict (serialized to JSON by the CLI) with two parts:

- ``aggregates``: mean of each metric across all queries — what the
  regression gate compares against the baseline.
- ``per_query``: every query's own scores — what you read when a metric
  drops and you need to know *which* questions got worse.
"""

from __future__ import annotations

from dataclasses import dataclass

from .datasets import QAExample
from .metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .retrieval import Retriever


@dataclass(frozen=True)
class QueryResult:
    qa_id: str
    question: str
    retrieved_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    scores: dict[str, float]


def evaluate_query(retriever: Retriever, example: QAExample, k: int) -> QueryResult:
    retrieved = retriever.retrieve(example.question, k)
    relevant = set(example.relevant_doc_ids)
    scores = {
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"hit_rate@{k}": hit_rate_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }
    return QueryResult(
        qa_id=example.id,
        question=example.question,
        retrieved_ids=tuple(retrieved),
        relevant_ids=example.relevant_doc_ids,
        scores=scores,
    )


def run_eval(retriever: Retriever, qa_set: list[QAExample], k: int) -> dict:
    if not qa_set:
        raise ValueError("QA set is empty")

    results = [evaluate_query(retriever, example, k) for example in qa_set]

    metric_names = results[0].scores.keys()
    aggregates = {
        name: sum(r.scores[name] for r in results) / len(results) for name in metric_names
    }

    return {
        "config": {"k": k, "num_queries": len(results)},
        "aggregates": aggregates,
        "per_query": [
            {
                "qa_id": r.qa_id,
                "question": r.question,
                "retrieved_ids": list(r.retrieved_ids),
                "relevant_ids": list(r.relevant_ids),
                "scores": r.scores,
            }
            for r in results
        ],
    }
