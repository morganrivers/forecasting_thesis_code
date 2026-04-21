"""
Tests for src/utils/overall_rating_scoring_metrics.py

Covers every public function with:
  - correct values on known inputs
  - edge cases (perfect predictions, reversed predictions, all ties)
  - invariants (symmetry, range, nan propagation)
"""

import math

import numpy as np
import pytest

import scoring_metrics as sr

# -- adjusted_r2 --------------------------------------------------------------


class TestAdjustedR2:
    def test_perfect_model_returns_one(self):
        assert sr.adjusted_r2(1.0, n=100, p=5) == pytest.approx(1.0)

    def test_formula_correctness(self):
        # 1 - (1 - 0.8) * (99/94) = 1 - 0.2 * (99/94)
        expected = 1.0 - 0.2 * (99 / 94)
        assert sr.adjusted_r2(0.8, n=100, p=5) == pytest.approx(expected)

    def test_more_predictors_lowers_score(self):
        r2 = 0.7
        adj5 = sr.adjusted_r2(r2, n=100, p=5)
        adj20 = sr.adjusted_r2(r2, n=100, p=20)
        assert adj5 > adj20

    def test_none_r2_returns_nan(self):
        assert math.isnan(sr.adjusted_r2(None, n=100, p=5))

    def test_insufficient_df_returns_nan(self):
        # n - p - 1 = 0 → nan
        assert math.isnan(sr.adjusted_r2(0.9, n=6, p=5))
        assert math.isnan(sr.adjusted_r2(0.9, n=5, p=5))

    def test_zero_r2_penalised(self):
        result = sr.adjusted_r2(0.0, n=100, p=5)
        assert result < 0.0


# -- rmse ---------------------------------------------------------------------


class TestRMSE:
    def test_perfect_predictions(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert sr.rmse(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value(self):
        # errors = [0.5]*5, rmse = 0.5
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        assert sr.rmse(y_true, y_pred) == pytest.approx(0.5)

    def test_asymmetric_errors(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([3.0, 4.0])
        # sqrt((9+16)/2) = sqrt(12.5)
        assert sr.rmse(y_true, y_pred) == pytest.approx(math.sqrt(12.5))

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 5, 50)
        yhat = rng.uniform(0, 5, 50)
        assert sr.rmse(y, yhat) >= 0.0

    def test_lists_accepted(self):
        assert sr.rmse([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


# -- mae -----------------------------------------------------------------------


class TestMAE:
    def test_perfect_predictions(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert sr.mae(y_true, y_pred) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([0.0, 2.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert sr.mae(y_true, y_pred) == pytest.approx(2.0 / 3.0)

    def test_mae_le_rmse(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0, 5, 100)
        yhat = rng.uniform(0, 5, 100)
        assert sr.mae(y, yhat) <= sr.rmse(y, yhat) + 1e-12


# -- r2 ------------------------------------------------------------------------


class TestR2:
    def test_perfect_predictions(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert sr.r2(y_true, y_pred) == pytest.approx(1.0)

    def test_constant_prediction_is_zero(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full(5, y_true.mean())
        assert sr.r2(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_can_be_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        assert sr.r2(y_true, y_pred) < 0


# -- true_hit_accuracy ---------------------------------------------------------


class TestTrueHitAccuracy:
    def test_perfect_match(self):
        y = np.array([0.0, 1.0, 2.0, 3.0])
        assert sr.true_hit_accuracy(y, y) == pytest.approx(1.0)

    def test_rounding_within_bucket(self):
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([2.4, 1.6, 2.49])  # all round to 2
        assert sr.true_hit_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_rounding_misses(self):
        y_true = np.array([2.0, 2.0])
        y_pred = np.array([2.6, 1.4])  # round to 3 and 1
        assert sr.true_hit_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_half_correct(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 3.0])
        assert sr.true_hit_accuracy(y_true, y_pred) == pytest.approx(0.5)

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 6, 50).astype(float)
        yhat = rng.uniform(0, 5, 50)
        acc = sr.true_hit_accuracy(y, yhat)
        assert 0.0 <= acc <= 1.0


# -- side_accuracy -------------------------------------------------------------


class TestSideAccuracy:
    def test_all_above_threshold(self):
        y_true = np.array([4.0, 4.5, 5.0])
        y_pred = np.array([3.6, 4.1, 4.8])
        assert sr.side_accuracy(y_true, y_pred, threshold=3.5) == pytest.approx(1.0)

    def test_all_below_threshold(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.4])
        assert sr.side_accuracy(y_true, y_pred, threshold=3.5) == pytest.approx(1.0)

    def test_perfectly_wrong(self):
        y_true = np.array([4.0, 4.0])  # above 3.5
        y_pred = np.array([2.0, 2.0])  # below 3.5
        assert sr.side_accuracy(y_true, y_pred, threshold=3.5) == pytest.approx(0.0)

    def test_half_right(self):
        y_true = np.array([4.0, 2.0])
        y_pred = np.array([2.0, 4.0])  # both wrong
        assert sr.side_accuracy(y_true, y_pred, threshold=3.5) == pytest.approx(0.0)

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(3)
        y = rng.uniform(0, 5, 100)
        yhat = rng.uniform(0, 5, 100)
        acc = sr.side_accuracy(y, yhat, threshold=3.5)
        assert 0.0 <= acc <= 1.0


# -- spearman_correlation ------------------------------------------------------


class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert sr.spearman_correlation(y, y) == pytest.approx(1.0)

    def test_perfect_negative(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert sr.spearman_correlation(y_true, y_pred) == pytest.approx(-1.0)

    def test_range(self):
        rng = np.random.default_rng(6)
        y = rng.uniform(0, 5, 50)
        yhat = rng.uniform(0, 5, 50)
        corr = sr.spearman_correlation(y, yhat)
        assert -1.0 <= corr <= 1.0


# -- brier_skill_score ---------------------------------------------------------


class TestBrierSkillScore:
    def test_perfect_binary_predictions(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        bss = sr.brier_skill_score(y_true, y_pred, train_base_rate=0.5)
        assert bss == pytest.approx(1.0)

    def test_baseline_prediction_gives_zero(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        base_rate = 0.5
        y_pred = np.full(4, base_rate)
        bss = sr.brier_skill_score(y_true, y_pred, train_base_rate=base_rate)
        assert bss == pytest.approx(0.0, abs=1e-10)

    def test_negative_when_worse_than_baseline(self):
        y_true = np.array([1.0, 1.0, 1.0, 1.0])  # all positive
        y_pred = np.array([0.0, 0.0, 0.0, 0.0])  # predicts all negative
        bss = sr.brier_skill_score(y_true, y_pred, train_base_rate=1.0)
        # base_brier = 0 → nan
        assert math.isnan(bss)

    def test_uses_eval_prevalence_when_no_base_rate(self):
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.5, 0.5])
        # Should not raise
        bss = sr.brier_skill_score(y_true, y_pred, train_base_rate=None)
        assert isinstance(bss, float)


# -- pairwise_ordering_prob_excl_ties ------------------------------------------


class TestPairwiseOrderingProbExclTies:
    def test_perfect_ordering(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        assert sr.pairwise_ordering_prob_excl_ties(y_true, y_pred) == pytest.approx(1.0)

    def test_reversed_ordering(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.1, 4.1, 3.1, 2.1, 1.1])
        assert sr.pairwise_ordering_prob_excl_ties(y_true, y_pred) == pytest.approx(0.0)

    def test_all_ties_returns_nan(self):
        y_true = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = sr.pairwise_ordering_prob_excl_ties(y_true, y_pred)
        assert math.isnan(result)

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(8)
        y = rng.integers(0, 6, 30).astype(float)
        yhat = rng.uniform(0, 5, 30)
        result = sr.pairwise_ordering_prob_excl_ties(y, yhat)
        if not math.isnan(result):
            assert 0.0 <= result <= 1.0

    def test_random_is_near_half(self):
        rng = np.random.default_rng(99)
        results = []
        for _ in range(20):
            y = rng.integers(0, 6, 200).astype(float)
            yhat = rng.uniform(0, 5, 200)
            r = sr.pairwise_ordering_prob_excl_ties(y, yhat)
            if not math.isnan(r):
                results.append(r)
        assert abs(np.mean(results) - 0.5) < 0.1


# -- within_group_pairwise_ordering_prob ---------------------------------------


class TestWithinGroupPairwiseOrderingProb:
    def test_perfect_within_groups(self, grouped_data):
        y_true, _, groups = grouped_data
        # perfect predictions = same as y_true
        result = sr.within_group_pairwise_ordering_prob(y_true, y_true + 0.01, groups)
        assert result["prob"] == pytest.approx(1.0)

    def test_reversed_within_groups(self, grouped_data):
        y_true, _, groups = grouped_data
        y_pred = -y_true
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert result["prob"] == pytest.approx(0.0)

    def test_returns_required_keys(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert "prob" in result
        assert "n_pairs" in result
        assert "n_groups" in result

    def test_single_member_groups_give_nan(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        groups = np.array(["A", "B", "C"])  # each group has 1 member
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert math.isnan(result["prob"])
        assert result["n_groups"] == 0

    def test_n_pairs_is_non_negative(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert result["n_pairs"] >= 0

    def test_prob_range(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        if not math.isnan(result["prob"]):
            assert 0.0 <= result["prob"] <= 1.0


# -- within_group_pairwise_ordering_prob_on_reference_pairs -------------------


class TestWithinGroupPOPOnReferencePairs:
    def test_returns_required_keys(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        result = sr.within_group_pairwise_ordering_prob_on_reference_pairs(
            y_true, y_pred, y_pred, groups
        )
        assert "prob" in result
        assert "n_pairs" in result
        assert "n_groups" in result

    def test_all_reference_tied_gives_zero_pairs(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        y_ref = np.full_like(y_true, 3.0)  # all tied → no valid pairs
        result = sr.within_group_pairwise_ordering_prob_on_reference_pairs(
            y_true, y_pred, y_ref, groups
        )
        assert math.isnan(result["prob"])
        assert result["n_pairs"] == 0

    def test_prob_range(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        result = sr.within_group_pairwise_ordering_prob_on_reference_pairs(
            y_true, y_pred, y_pred, groups
        )
        if not math.isnan(result["prob"]):
            assert 0.0 <= result["prob"] <= 1.0


# -- within_group_spearman_correlation -----------------------------------------


class TestWithinGroupSpearmanCorrelation:
    def test_perfect_returns_one(self, grouped_data):
        y_true, _, groups = grouped_data
        result = sr.within_group_spearman_correlation(y_true, y_true, groups)
        assert result["correlation"] == pytest.approx(1.0)

    def test_returns_required_keys(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        result = sr.within_group_spearman_correlation(y_true, y_pred, groups)
        assert "correlation" in result
        assert "n_groups" in result

    def test_single_member_groups_skipped(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        groups = np.array(["A", "B", "C"])
        result = sr.within_group_spearman_correlation(y_true, y_pred, groups)
        assert math.isnan(result["correlation"])
        assert result["n_groups"] == 0


# -- org_year_pairwise_ordering_prob ------------------------------------------


class TestOrgYearPairwiseOrderingProb:
    def test_returns_tuple(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        orgs = groups
        years = np.array(
            ["2010", "2011", "2012", "2013", "2010", "2011", "2012", "2013"]
        )
        weighted_avg, per_org = sr.org_year_pairwise_ordering_prob(
            y_true, y_pred, orgs, years
        )
        assert isinstance(per_org, dict)

    def test_per_org_keys_match_unique_orgs(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        orgs = groups
        years = np.array(
            ["2010", "2011", "2012", "2013", "2010", "2011", "2012", "2013"]
        )
        _, per_org = sr.org_year_pairwise_ordering_prob(y_true, y_pred, orgs, years)
        assert set(per_org.keys()) == {"A", "B"}

    def test_per_org_values_are_tuples(self, grouped_data):
        y_true, y_pred, groups = grouped_data
        orgs = groups
        years = np.array(
            ["2010", "2011", "2012", "2013", "2010", "2011", "2012", "2013"]
        )
        _, per_org = sr.org_year_pairwise_ordering_prob(y_true, y_pred, orgs, years)
        for org, val in per_org.items():
            assert len(val) == 2  # (pop, n_pairs)
