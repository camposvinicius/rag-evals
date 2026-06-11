"""Dataset loading: a document corpus, an annotated QA set, and optionally
precomputed retrieval results — all JSONL.

Formats (one JSON object per line):

corpus.jsonl     {"id": "doc-001", "title": "...", "text": "..."}
qa.jsonl         {"id": "q-001", "question": "...", "relevant_doc_ids": ["doc-001"]}
retrieved.jsonl  {"qa_id": "q-001", "retrieved_doc_ids": ["doc-007", "doc-001"]}

``relevant_doc_ids`` is the human annotation: which documents a correct
answer must come from. ``retrieved.jsonl`` lets any external RAG stack
(LangChain, Elasticsearch, pgvector, an internal API...) export its results
and be scored by this harness. The loaders validate aggressively — a
malformed dataset should fail loudly at load time, not skew metrics silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    id: str
    question: str
    relevant_doc_ids: tuple[str, ...]


class DatasetError(ValueError):
    """Raised when a dataset file is malformed."""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    if not rows:
        raise DatasetError(f"{path}: file is empty")
    return rows


def load_corpus(path: Path) -> list[Document]:
    docs, seen = [], set()
    for row in _read_jsonl(path):
        try:
            doc = Document(id=row["id"], title=row["title"], text=row["text"])
        except KeyError as exc:
            raise DatasetError(f"{path}: document missing field {exc}") from exc
        if not doc.text.strip():
            raise DatasetError(f"{path}: document {doc.id!r} has empty text")
        if doc.id in seen:
            raise DatasetError(f"{path}: duplicate document id {doc.id!r}")
        seen.add(doc.id)
        docs.append(doc)
    return docs


def load_qa(path: Path, corpus_ids: set[str] | None) -> list[QAExample]:
    """Load the annotated QA set.

    ``corpus_ids`` enables referential validation (every relevant doc id
    must exist in the corpus). Pass ``None`` only when no corpus is
    available — e.g. scoring precomputed results from an external stack —
    in which case id consistency is the caller's responsibility.
    """
    examples, seen = [], set()
    for row in _read_jsonl(path):
        try:
            example = QAExample(
                id=row["id"],
                question=row["question"],
                relevant_doc_ids=tuple(row["relevant_doc_ids"]),
            )
        except KeyError as exc:
            raise DatasetError(f"{path}: QA example missing field {exc}") from exc
        if not example.relevant_doc_ids:
            raise DatasetError(f"{path}: QA example {example.id!r} has no relevant_doc_ids")
        if corpus_ids is not None:
            unknown = set(example.relevant_doc_ids) - corpus_ids
            if unknown:
                raise DatasetError(
                    f"{path}: QA example {example.id!r} references unknown doc ids: "
                    f"{sorted(unknown)}"
                )
        if example.id in seen:
            raise DatasetError(f"{path}: duplicate QA id {example.id!r}")
        seen.add(example.id)
        examples.append(example)
    return examples


def load_retrieved(path: Path, qa_ids: set[str]) -> dict[str, list[str]]:
    """Load precomputed retrieval results keyed by QA id.

    Every QA example must have exactly one entry (an empty list is a valid
    result — the retriever found nothing — but a *missing* entry is a bug).
    """
    results: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        try:
            qa_id = row["qa_id"]
            retrieved = list(row["retrieved_doc_ids"])
        except KeyError as exc:
            raise DatasetError(f"{path}: retrieved entry missing field {exc}") from exc
        if qa_id in results:
            raise DatasetError(f"{path}: duplicate qa_id {qa_id!r}")
        if qa_id not in qa_ids:
            raise DatasetError(f"{path}: qa_id {qa_id!r} does not exist in the QA set")
        results[qa_id] = retrieved

    missing = qa_ids - results.keys()
    if missing:
        raise DatasetError(
            f"{path}: no retrieved entry for QA ids: {sorted(missing)}. "
            "Every annotated question must be present (an empty list is allowed)."
        )
    return results
