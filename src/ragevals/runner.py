"""Run an evaluation: retrieve for every QA example, score, aggregate.

The output is a plain dict (serialized to JSON by the CLI) with two parts:

- ``aggregates``: mean of each metric across all queries — what the
  regression gate compares against the baseline.
- ``per_query``: every query's own scores — what you read when a metric
  drops and you need to know *which* questions got worse.
"""

from __future__ import annotations

from dataclasses import dataclass

from .datasets import Document, QAExample
from .generation import generate_answer
from .metrics.judge import LLMClient, judge_faithfulness
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
    return score_retrieved(example, retrieved, k)


def score_retrieved(example: QAExample, retrieved: list[str], k: int) -> QueryResult:
    """Score an already-retrieved ranking for one query.

    Only the top-k of ``retrieved`` is considered, so externally produced
    rankings longer than k are scored exactly like internal ones.
    """
    retrieved = retrieved[:k]
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
    return _build_run_doc(results, k)


def run_eval_from_results(
    retrieved_by_qa_id: dict[str, list[str]], qa_set: list[QAExample], k: int
) -> dict:
    """Score precomputed retrieval results (the stack-agnostic entry point)."""
    if not qa_set:
        raise ValueError("QA set is empty")
    results = [
        score_retrieved(example, retrieved_by_qa_id[example.id], k) for example in qa_set
    ]
    return _build_run_doc(results, k)


def _build_run_doc(results: list[QueryResult], k: int) -> dict:
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


def run_generation_eval(
    run_doc: dict,
    documents: list[Document],
    generator: LLMClient,
    judge: LLMClient,
) -> dict:
    """Generate an answer per query from its retrieved docs, judge faithfulness.

    Mutates ``run_doc`` in place: adds ``generation_aggregates`` and a
    ``generation`` block per query. Generation metrics live in a separate
    key (not ``aggregates``) on purpose: the regression gate runs offline
    in CI without credentials, so judge metrics are informative, not gating.
    """
    docs_by_id = {d.id: d for d in documents}
    scores = []

    for query in run_doc["per_query"]:
        contexts = [docs_by_id[doc_id].text for doc_id in query["retrieved_ids"]]
        answer = generate_answer(generator, query["question"], contexts)
        verdict = judge_faithfulness(judge, query["question"], answer, contexts)
        query["generation"] = {
            "answer": answer,
            "faithfulness": verdict["score"],
            "unsupported_claims": verdict["unsupported_claims"],
        }
        scores.append(verdict["score"])

    run_doc["generation_aggregates"] = {"faithfulness": sum(scores) / len(scores)}
    return run_doc
