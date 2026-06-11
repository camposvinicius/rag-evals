"""Per-query diff between two runs.

The regression gate (``check``) answers "did quality drop?" on aggregates.
This module answers the follow-up question that actually matters when it
fails: **which queries got worse, and what did the retriever return?**
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryDelta:
    qa_id: str
    question: str
    metric: str
    baseline: float
    current: float
    baseline_retrieved: list[str]
    current_retrieved: list[str]
    relevant: list[str]

    @property
    def delta(self) -> float:
        return self.current - self.baseline


def diff_runs(baseline_doc: dict, run_doc: dict, metric: str) -> list[QueryDelta]:
    """Return per-query deltas for ``metric``, worst regressions first.

    Both runs must contain ``per_query`` (full run files, not baselines —
    baselines only keep aggregates) and must cover the same QA ids.
    """
    for name, doc in (("baseline", baseline_doc), ("run", run_doc)):
        if "per_query" not in doc:
            raise ValueError(
                f"{name} file has no per_query section — diff needs full run files, "
                "not baseline files (baselines only keep aggregates)"
            )

    baseline_by_id = {q["qa_id"]: q for q in baseline_doc["per_query"]}
    current_by_id = {q["qa_id"]: q for q in run_doc["per_query"]}

    if baseline_by_id.keys() != current_by_id.keys():
        only_baseline = sorted(baseline_by_id.keys() - current_by_id.keys())
        only_current = sorted(current_by_id.keys() - baseline_by_id.keys())
        raise ValueError(
            f"Runs cover different QA sets (only in baseline: {only_baseline}, "
            f"only in run: {only_current})"
        )

    sample = next(iter(baseline_by_id.values()))
    if metric not in sample["scores"]:
        available = ", ".join(sorted(sample["scores"]))
        raise ValueError(f"Unknown metric {metric!r}. Available: {available}")

    deltas = [
        QueryDelta(
            qa_id=qa_id,
            question=current["question"],
            metric=metric,
            baseline=baseline_by_id[qa_id]["scores"][metric],
            current=current["scores"][metric],
            baseline_retrieved=baseline_by_id[qa_id]["retrieved_ids"],
            current_retrieved=current["retrieved_ids"],
            relevant=current["relevant_ids"],
        )
        for qa_id, current in current_by_id.items()
    ]
    return sorted(deltas, key=lambda d: d.delta)
