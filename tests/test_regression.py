"""Tests for the regression gate."""

import pytest

from ragevals.regression import compare

BASELINE = {"recall@5": 0.90, "mrr": 0.80}


def test_identical_run_passes():
    result = compare(BASELINE, dict(BASELINE), tolerance=0.02)
    assert result.passed
    assert result.regressions == []


def test_drop_within_tolerance_passes():
    current = {"recall@5": 0.89, "mrr": 0.80}  # -0.01 <= 0.02
    assert compare(BASELINE, current, tolerance=0.02).passed


def test_drop_beyond_tolerance_fails():
    current = {"recall@5": 0.85, "mrr": 0.80}  # -0.05 > 0.02
    result = compare(BASELINE, current, tolerance=0.02)
    assert not result.passed
    assert [c.name for c in result.regressions] == ["recall@5"]


def test_improvement_never_fails_and_is_reported():
    current = {"recall@5": 0.97, "mrr": 0.80}
    result = compare(BASELINE, current, tolerance=0.02)
    assert result.passed
    assert [c.name for c in result.improvements] == ["recall@5"]


def test_metric_set_mismatch_is_an_error():
    current = {"recall@5": 0.90, "ndcg@5": 0.9}
    with pytest.raises(ValueError, match="different metrics"):
        compare(BASELINE, current, tolerance=0.02)


def test_zero_tolerance_fails_any_drop():
    current = {"recall@5": 0.8999, "mrr": 0.80}
    assert not compare(BASELINE, current, tolerance=0.0).passed
