"""Dataset loading: a document corpus and an annotated QA set, both JSONL.

Formats (one JSON object per line):

corpus.jsonl   {"id": "doc-001", "title": "...", "text": "..."}
qa.jsonl       {"id": "q-001", "question": "...", "relevant_doc_ids": ["doc-001"]}

``relevant_doc_ids`` is the human annotation: which documents a correct
answer must come from. The loaders validate aggressively — a malformed
dataset should fail loudly at load time, not skew metrics silently.
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


def load_qa(path: Path, corpus_ids: set[str]) -> list[QAExample]:
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
        unknown = set(example.relevant_doc_ids) - corpus_ids
        if unknown:
            raise DatasetError(
                f"{path}: QA example {example.id!r} references unknown doc ids: {sorted(unknown)}"
            )
        if example.id in seen:
            raise DatasetError(f"{path}: duplicate QA id {example.id!r}")
        seen.add(example.id)
        examples.append(example)
    return examples
