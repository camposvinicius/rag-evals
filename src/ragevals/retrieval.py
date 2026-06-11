"""Retrievers. The default (BM25) is fully offline so CI needs no API keys.

A retriever takes a query string and returns document ids in rank order.
Keeping this interface tiny means swapping BM25 for embeddings later is a
one-class change — the metrics and the runner never know the difference.
"""

from __future__ import annotations

import re
from typing import Protocol

from rank_bm25 import BM25Okapi

from .datasets import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer — simple, deterministic, dependency-free."""
    return _TOKEN_RE.findall(text.lower())


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]:
        """Return up to k document ids, best match first."""
        ...


class BM25Retriever:
    """Classic lexical ranking (BM25) over title + text.

    Chosen as the default because it is deterministic, fast, and needs no
    network — the eval harness itself must be cheap to run on every commit.
    """

    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("BM25Retriever needs at least one document")
        self._ids = [doc.id for doc in documents]
        tokenized = [tokenize(f"{doc.title} {doc.text}") for doc in documents]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._ids[i] for i in ranked[:k]]
