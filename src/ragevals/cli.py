"""Command-line interface.

Commands:
    ragevals run              — evaluate a corpus + QA set, write run.json
    ragevals check            — compare a run against the baseline (CI gate)
    ragevals update-baseline  — promote a run's aggregates to be the new baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .config import ConfigError, Settings
from .datasets import load_corpus, load_qa
from .regression import compare
from .retrieval import BM25Retriever
from .runner import run_eval, run_generation_eval


@click.group()
def main() -> None:
    """rag-evals: evaluation harness for RAG pipelines."""


@main.command()
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--qa", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("run.json"))
@click.option(
    "--with-generation",
    is_flag=True,
    help="Also generate answers and judge their faithfulness (requires judge config).",
)
def run(corpus: Path, qa: Path, output: Path, with_generation: bool) -> None:
    """Evaluate retrieval quality and write the results to OUTPUT."""
    settings = Settings.from_env()

    documents = load_corpus(corpus)
    qa_set = load_qa(qa, corpus_ids={d.id for d in documents})
    retriever = BM25Retriever(documents)

    result = run_eval(retriever, qa_set, k=settings.top_k)

    if with_generation:
        if not settings.judge_provider:
            raise ConfigError("--with-generation requires RAGEVALS_JUDGE_PROVIDER")
        if not settings.generation_model:
            raise ConfigError("--with-generation requires RAGEVALS_GENERATION_MODEL")
        from .metrics.judge import BedrockClient

        generator = BedrockClient(model_id=settings.generation_model)
        judge = BedrockClient(model_id=settings.judge_model)
        result = run_generation_eval(result, documents, generator, judge)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    click.echo(f"Evaluated {result['config']['num_queries']} queries (k={settings.top_k})")
    for name, value in result["aggregates"].items():
        click.echo(f"  {name:<14} {value:.4f}")
    if "generation_aggregates" in result:
        for name, value in result["generation_aggregates"].items():
            click.echo(f"  {name:<14} {value:.4f}")
    click.echo(f"Wrote {output}")


@main.command()
@click.option("--run", "run_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--baseline",
    type=click.Path(exists=True, path_type=Path),
    default=Path("baselines/baseline.json"),
)
def check(run_path: Path, baseline: Path) -> None:
    """Fail (exit 1) if RUN regressed against BASELINE beyond the tolerance."""
    settings = Settings.from_env()

    run_doc = json.loads(run_path.read_text())
    baseline_doc = json.loads(baseline.read_text())

    result = compare(
        baseline_doc["aggregates"], run_doc["aggregates"], settings.regression_tolerance
    )

    click.echo(f"Regression gate (tolerance: {result.tolerance})")
    for c in result.comparisons:
        marker = "OK "
        if c in result.regressions:
            marker = "REG"
        elif c in result.improvements:
            marker = "IMP"
        click.echo(f"  [{marker}] {c.name:<14} baseline={c.baseline:.4f} "
                   f"current={c.current:.4f} delta={c.delta:+.4f}")

    if result.improvements:
        click.echo("Improvements detected — consider `ragevals update-baseline` to lock them in.")

    if not result.passed:
        click.echo("FAILED: metrics regressed beyond tolerance.", err=True)
        sys.exit(1)
    click.echo("PASSED")


@main.command(name="update-baseline")
@click.option("--run", "run_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--baseline", type=click.Path(path_type=Path), default=Path("baselines/baseline.json")
)
def update_baseline(run_path: Path, baseline: Path) -> None:
    """Promote RUN's aggregates to be the new committed baseline."""
    run_doc = json.loads(run_path.read_text())
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        json.dumps({"config": run_doc["config"], "aggregates": run_doc["aggregates"]}, indent=2)
        + "\n"
    )
    click.echo(f"Baseline updated: {baseline}")
