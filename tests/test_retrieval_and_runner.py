"""Tests for the BM25 retriever and the eval runner."""

import pytest

from ragevals.datasets import Document, QAExample
from ragevals.retrieval import BM25Retriever, tokenize
from ragevals.runner import run_eval

DOCS = [
    Document(id="d-auth", title="Authentication", text="API keys authenticate every request."),
    Document(id="d-rate", title="Rate limits", text="Requests above the limit get HTTP 429."),
    Document(id="d-bill", title="Billing", text="Compute is billed per second of usage."),
]


def test_tokenize_is_lowercase_alnum():
    assert tokenize("HTTP 429, Retry-After!") == ["http", "429", "retry", "after"]


def test_bm25_ranks_the_obvious_doc_first():
    retriever = BM25Retriever(DOCS)
    assert retriever.retrieve("how does authentication work", k=1) == ["d-auth"]
    assert retriever.retrieve("what is the rate limit", k=1) == ["d-rate"]


def test_bm25_respects_k():
    retriever = BM25Retriever(DOCS)
    assert len(retriever.retrieve("billing", k=2)) == 2


def test_run_eval_aggregates_are_means():
    retriever = BM25Retriever(DOCS)
    qa = [
        QAExample(id="q1", question="authentication api keys", relevant_doc_ids=("d-auth",)),
        QAExample(id="q2", question="rate limit 429", relevant_doc_ids=("d-rate",)),
    ]
    result = run_eval(retriever, qa, k=1)

    assert result["config"]["num_queries"] == 2
    # BM25 should nail both queries at rank 1 -> every metric averages to 1.0
    for name, value in result["aggregates"].items():
        assert value == pytest.approx(1.0), name


def test_run_eval_empty_qa_set_fails():
    retriever = BM25Retriever(DOCS)
    with pytest.raises(ValueError, match="QA set is empty"):
        run_eval(retriever, [], k=1)
