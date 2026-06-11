"""Tests for the stack-agnostic mode (--retrieved) and per-query diff."""

import pytest

from ragevals.datasets import DatasetError, QAExample, load_retrieved
from ragevals.diff import diff_runs
from ragevals.runner import run_eval_from_results

QA = [
    QAExample(id="q1", question="What is X?", relevant_doc_ids=("d1",)),
    QAExample(id="q2", question="What is Y?", relevant_doc_ids=("d2",)),
]


class TestLoadRetrieved:
    def test_ok(self, tmp_path):
        path = tmp_path / "retrieved.jsonl"
        path.write_text(
            '{"qa_id": "q1", "retrieved_doc_ids": ["d1", "d3"]}\n'
            '{"qa_id": "q2", "retrieved_doc_ids": []}\n'
        )
        results = load_retrieved(path, qa_ids={"q1", "q2"})
        assert results == {"q1": ["d1", "d3"], "q2": []}

    def test_missing_qa_entry_fails(self, tmp_path):
        path = tmp_path / "retrieved.jsonl"
        path.write_text('{"qa_id": "q1", "retrieved_doc_ids": ["d1"]}\n')
        with pytest.raises(DatasetError, match="no retrieved entry for QA ids"):
            load_retrieved(path, qa_ids={"q1", "q2"})

    def test_unknown_qa_id_fails(self, tmp_path):
        path = tmp_path / "retrieved.jsonl"
        path.write_text('{"qa_id": "q999", "retrieved_doc_ids": ["d1"]}\n')
        with pytest.raises(DatasetError, match="does not exist in the QA set"):
            load_retrieved(path, qa_ids={"q1"})

    def test_duplicate_qa_id_fails(self, tmp_path):
        path = tmp_path / "retrieved.jsonl"
        path.write_text(
            '{"qa_id": "q1", "retrieved_doc_ids": ["d1"]}\n'
            '{"qa_id": "q1", "retrieved_doc_ids": ["d2"]}\n'
        )
        with pytest.raises(DatasetError, match="duplicate qa_id"):
            load_retrieved(path, qa_ids={"q1"})


class TestRunEvalFromResults:
    def test_scores_external_results(self):
        results = {"q1": ["d1"], "q2": ["d9"]}  # q1 perfect, q2 miss
        doc = run_eval_from_results(results, QA, k=1)
        assert doc["aggregates"]["recall@1"] == 0.5
        assert doc["per_query"][0]["scores"]["recall@1"] == 1.0
        assert doc["per_query"][1]["scores"]["recall@1"] == 0.0

    def test_only_top_k_is_scored(self):
        # relevant doc at position 2, k=1 -> miss, even though it was returned
        results = {"q1": ["d9", "d1"], "q2": ["d2"]}
        doc = run_eval_from_results(results, QA, k=1)
        assert doc["per_query"][0]["scores"]["recall@1"] == 0.0


def _run_doc(scores_q1: float, scores_q2: float, retrieved_q1=("d9",)) -> dict:
    return {
        "config": {"k": 1, "num_queries": 2},
        "aggregates": {"mrr": (scores_q1 + scores_q2) / 2},
        "per_query": [
            {
                "qa_id": "q1",
                "question": "What is X?",
                "retrieved_ids": list(retrieved_q1),
                "relevant_ids": ["d1"],
                "scores": {"mrr": scores_q1},
            },
            {
                "qa_id": "q2",
                "question": "What is Y?",
                "retrieved_ids": ["d2"],
                "relevant_ids": ["d2"],
                "scores": {"mrr": scores_q2},
            },
        ],
    }


class TestDiffRuns:
    def test_worst_regression_first_with_context(self):
        baseline = _run_doc(1.0, 1.0, retrieved_q1=("d1",))
        current = _run_doc(0.0, 1.0, retrieved_q1=("d9",))
        deltas = diff_runs(baseline, current, "mrr")
        worst = deltas[0]
        assert worst.qa_id == "q1"
        assert worst.delta == -1.0
        assert worst.baseline_retrieved == ["d1"]
        assert worst.current_retrieved == ["d9"]
        assert worst.relevant == ["d1"]

    def test_unknown_metric_lists_available(self):
        with pytest.raises(ValueError, match="Available: mrr"):
            diff_runs(_run_doc(1, 1), _run_doc(1, 1), "recall@9")

    def test_baseline_file_without_per_query_is_explained(self):
        with pytest.raises(ValueError, match="baselines only keep aggregates"):
            diff_runs({"aggregates": {}}, _run_doc(1, 1), "mrr")

    def test_different_qa_sets_fail(self):
        current = _run_doc(1, 1)
        current["per_query"] = current["per_query"][:1]
        with pytest.raises(ValueError, match="different QA sets"):
            diff_runs(_run_doc(1, 1), current, "mrr")
