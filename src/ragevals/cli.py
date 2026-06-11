"""Command-line interface.

Commands:
    ragevals run    — evaluate a corpus + QA set, write run.json
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .config import Settings
from .datasets import load_corpus, load_qa
from .retrieval import BM25Retriever
from .runner import run_eval


@click.group()
def main() -> None:
    """rag-evals: evaluation harness for RAG pipelines."""


@main.command()
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--qa", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("run.json"))
def run(corpus: Path, qa: Path, output: Path) -> None:
    """Evaluate retrieval quality and write the results to OUTPUT."""
    settings = Settings.from_env()

    documents = load_corpus(corpus)
    qa_set = load_qa(qa, corpus_ids={d.id for d in documents})
    retriever = BM25Retriever(documents)

    result = run_eval(retriever, qa_set, k=settings.top_k)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    click.echo(f"Evaluated {result['config']['num_queries']} queries (k={settings.top_k})")
    for name, value in result["aggregates"].items():
        click.echo(f"  {name:<14} {value:.4f}")
    click.echo(f"Wrote {output}")
