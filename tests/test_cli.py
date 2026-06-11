"""CLI-level tests with Click's test runner.

These exercise the public interface end to end (arguments, wiring, error
messages) — the unit suites already cover the underlying modules.
"""

import json

from click.testing import CliRunner

from ragevals.cli import main

CORPUS = (
    '{"id": "d1", "title": "Auth", "text": "API keys authenticate requests."}\n'
    '{"id": "d2", "title": "Limits", "text": "Requests above the limit get HTTP 429."}\n'
)
QA = (
    '{"id": "q1", "question": "How do API keys work?", "relevant_doc_ids": ["d1"]}\n'
    '{"id": "q2", "question": "What happens above the rate limit?", "relevant_doc_ids": ["d2"]}\n'
)
RETRIEVED = (
    '{"qa_id": "q1", "retrieved_doc_ids": ["d1", "d2"]}\n'
    '{"qa_id": "q2", "retrieved_doc_ids": ["d1"]}\n'
)


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_run_with_builtin_bm25(tmp_path):
    corpus = _write(tmp_path, "corpus.jsonl", CORPUS)
    qa = _write(tmp_path, "qa.jsonl", QA)
    out = tmp_path / "run.json"

    result = CliRunner().invoke(
        main, ["run", "--corpus", str(corpus), "--qa", str(qa), "--output", str(out)]
    )

    assert result.exit_code == 0, result.output
    doc = json.loads(out.read_text())
    assert doc["config"]["num_queries"] == 2
    assert "recall@5" in doc["aggregates"]


def test_run_with_retrieved_only(tmp_path):
    qa = _write(tmp_path, "qa.jsonl", QA)
    retrieved = _write(tmp_path, "retrieved.jsonl", RETRIEVED)
    out = tmp_path / "run.json"

    result = CliRunner().invoke(
        main, ["run", "--qa", str(qa), "--retrieved", str(retrieved), "--output", str(out)]
    )

    assert result.exit_code == 0, result.output
    doc = json.loads(out.read_text())
    # q1 hit at rank 1 (mrr 1.0), q2 miss (mrr 0.0) -> mean 0.5
    assert doc["aggregates"]["mrr"] == 0.5


def test_run_without_corpus_or_retrieved_fails_cleanly(tmp_path):
    qa = _write(tmp_path, "qa.jsonl", QA)

    result = CliRunner().invoke(main, ["run", "--qa", str(qa)])

    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "--corpus" in result.output and "--retrieved" in result.output
    assert "Traceback" not in result.output


def test_run_with_generation_requires_corpus(tmp_path):
    qa = _write(tmp_path, "qa.jsonl", QA)
    retrieved = _write(tmp_path, "retrieved.jsonl", RETRIEVED)

    result = CliRunner().invoke(
        main,
        ["run", "--qa", str(qa), "--retrieved", str(retrieved), "--with-generation"],
    )

    assert result.exit_code != 0
    assert "requires --corpus" in result.output
    assert "Traceback" not in result.output


def test_check_passes_and_fails(tmp_path):
    corpus = _write(tmp_path, "corpus.jsonl", CORPUS)
    qa = _write(tmp_path, "qa.jsonl", QA)
    out = tmp_path / "run.json"
    baseline = tmp_path / "baseline.json"
    runner = CliRunner()

    runner.invoke(main, ["run", "--corpus", str(corpus), "--qa", str(qa), "--output", str(out)])
    runner.invoke(main, ["update-baseline", "--run", str(out), "--baseline", str(baseline)])

    ok = runner.invoke(main, ["check", "--run", str(out), "--baseline", str(baseline)])
    assert ok.exit_code == 0 and "PASSED" in ok.output

    doc = json.loads(out.read_text())
    doc["aggregates"]["mrr"] -= 0.5
    regressed = tmp_path / "regressed.json"
    regressed.write_text(json.dumps(doc))

    bad = runner.invoke(main, ["check", "--run", str(regressed), "--baseline", str(baseline)])
    assert bad.exit_code == 1 and "FAILED" in bad.output


def test_diff_shows_worst_regression_with_context(tmp_path):
    qa = _write(tmp_path, "qa.jsonl", QA)
    retrieved = _write(tmp_path, "retrieved.jsonl", RETRIEVED)
    old = tmp_path / "old.json"
    runner = CliRunner()
    runner.invoke(
        main, ["run", "--qa", str(qa), "--retrieved", str(retrieved), "--output", str(old)]
    )

    doc = json.loads(old.read_text())
    q1 = next(q for q in doc["per_query"] if q["qa_id"] == "q1")
    q1["scores"]["mrr"] = 0.0
    q1["retrieved_ids"] = ["d2"]
    new = tmp_path / "new.json"
    new.write_text(json.dumps(doc))

    result = runner.invoke(
        main, ["diff", "--baseline", str(old), "--run", str(new), "--metric", "mrr"]
    )

    assert result.exit_code == 0, result.output
    assert "Worst regressions" in result.output
    assert "q1" in result.output
    assert "relevant:  d1" in result.output


def test_diff_against_baseline_file_explains_the_mistake(tmp_path):
    qa = _write(tmp_path, "qa.jsonl", QA)
    retrieved = _write(tmp_path, "retrieved.jsonl", RETRIEVED)
    out = tmp_path / "run.json"
    baseline = tmp_path / "baseline.json"
    runner = CliRunner()
    runner.invoke(
        main, ["run", "--qa", str(qa), "--retrieved", str(retrieved), "--output", str(out)]
    )
    runner.invoke(main, ["update-baseline", "--run", str(out), "--baseline", str(baseline)])

    result = runner.invoke(
        main, ["diff", "--baseline", str(baseline), "--run", str(out)]
    )

    assert result.exit_code != 0
    assert "baselines only keep aggregates" in result.output
    assert "Traceback" not in result.output
