"""Grounded answer generation, used to produce the answers the judge scores.

This is intentionally a minimal RAG prompt — the project is an eval
harness, not a RAG framework. The generation step exists so the judge has
real answers to evaluate end-to-end.
"""

from __future__ import annotations

from .metrics.judge import LLMClient

_GENERATION_PROMPT = """\
Answer the question using ONLY the source documents below. If the documents
do not contain the answer, say "The provided documents do not answer this."
Do not use any outside knowledge.

Source documents:
{contexts}

Question: {question}

Answer:"""


def generate_answer(client: LLMClient, question: str, contexts: list[str]) -> str:
    prompt = _GENERATION_PROMPT.format(
        contexts="\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)),
        question=question,
    )
    return client.complete(prompt).strip()
