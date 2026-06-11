"""Hand-checkable tests for retrieval metrics.

The canonical scenario used in most tests:

    retrieved (in rank order): [d1, d2, d3, d4, d5]
    relevant:                  {d2, d9}

So: one relevant doc found (d2, at rank 2), one missed (d9).
"""

import math

import pytest

from ragevals.metrics.retrieval import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d9"}


class TestRecall:
    def test_finds_one_of_two(self):
        # found d2, missed d9 -> 1/2
        assert recall_at_k(RETRIEVED, RELEVANT, k=5) == 0.5

    def test_k_cuts_off_the_hit(self):
        # top-1 is [d1], no relevant docs -> 0
        assert recall_at_k(RETRIEVED, RELEVANT, k=1) == 0.0

    def test_perfect(self):
        assert recall_at_k(["d2", "d9"], RELEVANT, k=2) == 1.0


class TestPrecision:
    def test_one_hit_in_five(self):
        # 1 relevant in a budget of 5 -> 1/5
        assert precision_at_k(RETRIEVED, RELEVANT, k=5) == 0.2

    def test_denominator_is_k_not_len(self):
        # only 2 docs returned but budget is 5 -> 1/5, not 1/2
        assert precision_at_k(["d2", "d1"], RELEVANT, k=5) == 0.2


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k(RETRIEVED, RELEVANT, k=5) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(["d1", "d3"], RELEVANT, k=2) == 0.0


class TestReciprocalRank:
    def test_first_relevant_at_rank_2(self):
        assert reciprocal_rank(RETRIEVED, RELEVANT) == 0.5

    def test_relevant_at_rank_1(self):
        assert reciprocal_rank(["d9", "d1"], RELEVANT) == 1.0

    def test_no_relevant_found(self):
        assert reciprocal_rank(["d1", "d3"], RELEVANT) == 0.0


class TestNdcg:
    def test_single_hit_at_rank_2(self):
        # DCG  = 1/log2(2+1)                  (d2 at rank 2)
        # IDCG = 1/log2(2) + 1/log2(3)        (ideal: 2 relevant docs at ranks 1, 2)
        dcg = 1 / math.log2(3)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(RETRIEVED, RELEVANT, k=5) == pytest.approx(dcg / idcg)

    def test_perfect_ordering_is_1(self):
        assert ndcg_at_k(["d2", "d9", "d1"], RELEVANT, k=3) == pytest.approx(1.0)

    def test_rank_matters(self):
        # same single hit, earlier rank -> higher score
        early = ndcg_at_k(["d2", "d1", "d3"], RELEVANT, k=3)
        late = ndcg_at_k(["d1", "d3", "d2"], RELEVANT, k=3)
        assert early > late


class TestValidation:
    def test_empty_relevant_set_is_a_dataset_bug(self):
        with pytest.raises(ValueError, match="relevant_ids is empty"):
            recall_at_k(RETRIEVED, set(), k=5)

    def test_non_positive_k(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            recall_at_k(RETRIEVED, RELEVANT, k=0)
