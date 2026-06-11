"""Regression gate: compare a run against a committed baseline.

The contract is intentionally simple:

- The baseline (``baseline.json``) is a previous run's aggregates,
  committed to the repository like a lockfile.
- A new run **fails** if any metric drops more than ``tolerance`` below
  the baseline (absolute difference, e.g. 0.02 = 2 percentage points).
- Improvements never fail — but they are reported, so you remember to
  update the baseline and lock in the gain.

Why absolute tolerance instead of relative? Metrics here live in [0, 1].
A relative rule like "5% worse" gets stricter as scores get lower
(0.20 -> 0.19 fails) and looser as they get higher, which is backwards:
high-scoring suites are exactly where small drops matter most.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricComparison:
    name: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline


@dataclass(frozen=True)
class GateResult:
    comparisons: list[MetricComparison]
    tolerance: float

    @property
    def regressions(self) -> list[MetricComparison]:
        return [c for c in self.comparisons if c.delta < -self.tolerance]

    @property
    def improvements(self) -> list[MetricComparison]:
        return [c for c in self.comparisons if c.delta > self.tolerance]

    @property
    def passed(self) -> bool:
        return not self.regressions


def compare(baseline_aggregates: dict, current_aggregates: dict, tolerance: float) -> GateResult:
    """Compare aggregates; raise if the two runs measured different metrics."""
    baseline_keys = set(baseline_aggregates)
    current_keys = set(current_aggregates)
    if baseline_keys != current_keys:
        missing = baseline_keys - current_keys
        extra = current_keys - baseline_keys
        raise ValueError(
            "Baseline and run measure different metrics "
            f"(missing: {sorted(missing)}, extra: {sorted(extra)}). "
            "If the change is intentional, update the baseline."
        )

    comparisons = [
        MetricComparison(name=name, baseline=baseline_aggregates[name], current=value)
        for name, value in sorted(current_aggregates.items())
    ]
    return GateResult(comparisons=comparisons, tolerance=tolerance)
