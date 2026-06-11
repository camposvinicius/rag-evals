#!/usr/bin/env python3
"""Export your own RAG stack's retrieval results for rag-evals.

This example produces ``retrieved.jsonl`` — one line per annotated question:

    {"qa_id": "q-001", "retrieved_doc_ids": ["doc-002", "doc-001"]}

Replace ``my_retriever`` with whatever you already run in production
(LangChain, LlamaIndex, Elasticsearch, pgvector, OpenSearch, an internal
API...). rag-evals only needs the ranked ids:

    ragevals run --qa qa.jsonl --retrieved retrieved.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def my_retriever(question: str, k: int) -> list[str]:
    """Replace this with a call to your real retrieval stack."""
    raise NotImplementedError(
        "Wire this to your stack, e.g.:\n"
        "  return [hit.id for hit in es.search(question, size=k)]\n"
        "  return [d.metadata['id'] for d in vectorstore.similarity_search(question, k=k)]"
    )


def main(qa_path: Path, output_path: Path, k: int) -> None:
    with output_path.open("w") as out:
        for line in qa_path.read_text().splitlines():
            if not line.strip():
                continue
            example = json.loads(line)
            retrieved = my_retriever(example["question"], k)
            out.write(
                json.dumps({"qa_id": example["id"], "retrieved_doc_ids": retrieved}) + "\n"
            )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: export_retrieved.py <qa.jsonl> <retrieved.jsonl> <k>")
        raise SystemExit(2)
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
