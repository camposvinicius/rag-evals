"""Command-line interface.

Commands:
    ragevals run              — evaluate retrieval (built-in BM25 or external
                                results via --retrieved), write run.json
    ragevals check            — compare a run against the baseline (CI gate)
    ragevals diff             — per-query comparison of two runs
    ragevals update-baseline  — promote a run's aggregates to be the new baseline
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import click

from .config import ConfigError, Settings
from .datasets import DatasetError, load_corpus, load_qa, load_retrieved
from .diff import diff_runs
from .regression import compare
from .retrieval import BM25Retriever
from .runner import run_eval, run_eval_from_results, run_generation_eval


def friendly_errors(func):
    """Convert known domain errors into clean CLI messages (no tracebacks)."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ConfigError, DatasetError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    return wrapper


@click.group()
def main() -> None:
    """rag-evals: evaluation harness for RAG pipelines."""


@main.command()
@click.option("--corpus", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--qa", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--retrieved",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Score precomputed results ({qa_id, retrieved_doc_ids} JSONL) from any "
    "external RAG stack instead of running the built-in retriever.",
)
@click.option("--output", type=click.Path(path_type=Path), default=Path("run.json"))
@click.option(
    "--with-generation",
    is_flag=True,
    help="Also generate answers and judge their faithfulness (requires judge config).",
)
@friendly_errors
def run(
    corpus: Path | None, qa: Path, retrieved: Path | None, output: Path, with_generation: bool
) -> None:
    """Evaluate retrieval quality and write the results to OUTPUT."""
    settings = Settings.from_env()

    if corpus is None and retrieved is None:
        raise ConfigError("provide --corpus (built-in retrieval) and/or --retrieved "
                          "(precomputed results)")
    if with_generation and corpus is None:
        raise ConfigError("--with-generation requires --corpus (document texts are "
                          "needed to build the grounding context)")

    documents = load_corpus(corpus) if corpus else []
    corpus_ids = {d.id for d in documents} if documents else None
    qa_set = load_qa(qa, corpus_ids=corpus_ids)

    if retrieved is not None:
        results_by_id = load_retrieved(retrieved, qa_ids={e.id for e in qa_set})
        result = run_eval_from_results(results_by_id, qa_set, k=settings.top_k)
    else:
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
@friendly_errors
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
@friendly_errors
def update_baseline(run_path: Path, baseline: Path) -> None:
    """Promote RUN's aggregates to be the new committed baseline."""
    run_doc = json.loads(run_path.read_text())
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(
        json.dumps({"config": run_doc["config"], "aggregates": run_doc["aggregates"]}, indent=2)
        + "\n"
    )
    click.echo(f"Baseline updated: {baseline}")


@main.command()
@click.option(
    "--baseline", type=click.Path(exists=True, path_type=Path), required=True,
    help="A previous full run file (run.json), not a baseline file.",
)
@click.option("--run", "run_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--metric", default="mrr", show_default=True)
@click.option("--limit", default=5, show_default=True, help="How many queries to show.")
@friendly_errors
def diff(baseline: Path, run_path: Path, metric: str, limit: int) -> None:
    """Per-query comparison of two runs: which queries got worse, and why."""
    baseline_doc = json.loads(baseline.read_text())
    run_doc = json.loads(run_path.read_text())

    deltas = diff_runs(baseline_doc, run_doc, metric)
    changed = [d for d in deltas if d.delta != 0]

    if not changed:
        click.echo(f"No per-query changes in {metric}.")
        return

    regressions = [d for d in changed if d.delta < 0]
    improvements = [d for d in changed if d.delta > 0]

    if regressions:
        click.echo(f"Worst regressions ({metric}):")
        for d in regressions[:limit]:
            click.echo(f"  {d.qa_id}  {d.baseline:.4f} -> {d.current:.4f}  ({d.delta:+.4f})")
            click.echo(f"      Q: {d.question}")
            click.echo(f"      relevant:  {', '.join(d.relevant)}")
            click.echo(f"      baseline retrieved: {', '.join(d.baseline_retrieved)}")
            click.echo(f"      current retrieved:  {', '.join(d.current_retrieved)}")
    if improvements:
        click.echo(f"Improvements ({metric}):")
        for d in sorted(improvements, key=lambda d: -d.delta)[:limit]:
            click.echo(f"  {d.qa_id}  {d.baseline:.4f} -> {d.current:.4f}  ({d.delta:+.4f})")
