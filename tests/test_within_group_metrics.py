"""
Unit tests for within-group and ranking metrics in
src/utils/overall_rating_scoring_metrics.py.

Functions covered:
  - brier_skill_score
  - spearman_correlation
  - within_group_pairwise_ordering_prob
  - within_group_spearman_correlation
  - pairwise_ordering_prob_excl_ties
  - org_year_pairwise_ordering_prob
"""

import math

import numpy as np
import pytest

import scoring_metrics as sr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_groups(*sizes, prefix="G"):
    """Return a 1-D array of group labels with the given per-group sizes."""
    parts = []
    for k, n in enumerate(sizes):
        parts.extend([f"{prefix}{k}"] * n)
    return np.array(parts)


# ---------------------------------------------------------------------------
# brier_skill_score
# ---------------------------------------------------------------------------


class TestBrierSkillScore:
    """BSS = 1 - BS / BS_ref; ref uses constant base-rate predictor."""

    # --- perfect predictions → BSS = 1.0 ---
    @pytest.mark.parametrize("base_rate", [0.3, 0.5, 0.7])
    def test_perfect_gives_one(self, base_rate):
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        bss = sr.brier_skill_score(y, y.copy(), train_base_rate=base_rate)
        assert bss == pytest.approx(1.0)

    # --- predicting exactly the base rate → BSS = 0.0 ---
    @pytest.mark.parametrize("base_rate", [0.2, 0.5, 0.8])
    def test_baseline_predictor_gives_zero(self, base_rate):
        y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        y_pred = np.full(len(y_true), base_rate)
        bss = sr.brier_skill_score(y_true, y_pred, train_base_rate=base_rate)
        assert bss == pytest.approx(0.0, abs=1e-10)

    # --- worse than base rate → negative BSS ---
    def test_worse_than_baseline_is_negative(self):
        # base_rate = 0.5; predicting wrong labels has BS > BS_ref
        y_true = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        # predict 0 for all 1s and 1 for all 0s
        y_pred = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        bss = sr.brier_skill_score(y_true, y_pred, train_base_rate=0.5)
        assert bss < 0.0

    # --- degenerate base_rate = 1.0 with all-ones true → BS_ref = 0 → NaN ---
    def test_degenerate_base_rate_returns_nan(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([0.5, 0.5, 0.5])
        bss = sr.brier_skill_score(y_true, y_pred, train_base_rate=1.0)
        assert math.isnan(bss)

    # --- None base_rate falls back to eval prevalence ---
    def test_none_base_rate_uses_eval_prevalence(self):
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        # perfect prediction should still give BSS = 1 even without base_rate
        bss = sr.brier_skill_score(y_true, y_true.copy(), train_base_rate=None)
        assert bss == pytest.approx(1.0)

    # --- BSS is bounded above by 1.0 for any reasonable input ---
    def test_upper_bound_is_one(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 50).astype(float)
        y_pred = rng.uniform(0, 1, 50)
        bss = sr.brier_skill_score(y_true, y_pred, train_base_rate=0.5)
        assert bss <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# spearman_correlation
# ---------------------------------------------------------------------------


class TestSpearmanCorrelation:
    """Spearman rank correlation; range [-1, 1]."""

    @pytest.mark.parametrize("n", [5, 20, 100])
    def test_identical_arrays_give_one(self, n):
        y = np.arange(n, dtype=float)
        assert sr.spearman_correlation(y, y) == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [4, 10])
    def test_reversed_gives_minus_one(self, n):
        y = np.arange(n, dtype=float)
        assert sr.spearman_correlation(y, y[::-1]) == pytest.approx(-1.0)

    def test_uncorrelated_is_near_zero(self):
        """Uniformly random pairs have E[ρ]=0; 1000 obs is very tight."""
        rng = np.random.default_rng(0)
        y_true = rng.uniform(0, 5, 1000)
        y_pred = rng.uniform(0, 5, 1000)
        corr = sr.spearman_correlation(y_true, y_pred)
        assert abs(corr) < 0.1

    def test_output_in_range(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0, 5, 50)
        yhat = rng.uniform(0, 5, 50)
        corr = sr.spearman_correlation(y, yhat)
        assert -1.0 <= corr <= 1.0

    def test_returns_float(self):
        y = np.array([1.0, 2.0, 3.0])
        result = sr.spearman_correlation(y, y)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# pairwise_ordering_prob_excl_ties
# ---------------------------------------------------------------------------


class TestPairwiseOrderingProbExclTies:
    """Excludes tied y_true pairs AND tied y_pred pairs from denominator."""

    def test_perfect_order_gives_one(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert sr.pairwise_ordering_prob_excl_ties(y_true, y_pred) == pytest.approx(1.0)

    def test_reversed_order_gives_zero(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert sr.pairwise_ordering_prob_excl_ties(y_true, y_pred) == pytest.approx(0.0)

    def test_all_true_tied_returns_nan(self):
        """No non-tied true pairs → undefined → NaN."""
        y_true = np.full(5, 3.0)
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sr.pairwise_ordering_prob_excl_ties(y_true, y_pred)
        assert math.isnan(result)

    def test_all_pred_tied_returns_nan(self):
        """All predicted values identical → no non-tied pred pairs → NaN."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full(4, 2.5)
        result = sr.pairwise_ordering_prob_excl_ties(y_true, y_pred)
        assert math.isnan(result)

    def test_two_element_perfect(self):
        """Minimal case: two elements, correct order."""
        result = sr.pairwise_ordering_prob_excl_ties(
            np.array([1.0, 2.0]), np.array([0.5, 1.5])
        )
        assert result == pytest.approx(1.0)

    def test_two_element_reversed(self):
        result = sr.pairwise_ordering_prob_excl_ties(
            np.array([1.0, 2.0]), np.array([2.0, 1.0])
        )
        assert result == pytest.approx(0.0)

    def test_partial_correct_known_value(self):
        """
        y_true = [1, 2, 3], y_pred = [1, 3, 2]
        Pairs (all non-tied in both):
          (1,2): true diff=-1, pred diff=-2 → concordant
          (1,3): true diff=-2, pred diff=-1 → concordant
          (2,3): true diff=-1, pred diff=+1 → discordant
        Result = 2/3
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 3.0, 2.0])
        result = sr.pairwise_ordering_prob_excl_ties(y_true, y_pred)
        assert result == pytest.approx(2.0 / 3.0)

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_range_zero_to_one(self, seed):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 6, 40).astype(float)
        y_pred = rng.uniform(0, 5, 40)
        result = sr.pairwise_ordering_prob_excl_ties(y_true, y_pred)
        if not math.isnan(result):
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# within_group_pairwise_ordering_prob
# ---------------------------------------------------------------------------


class TestWithinGroupPairwiseOrderingProb:
    """Pairwise ordering prob stratified by group; 0.5 = random chance."""

    def test_perfect_ordering_two_groups(self):
        y_true = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        y_pred = np.array([1.1, 2.1, 3.1, 10.1, 20.1, 30.1])
        groups = _make_groups(3, 3)
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert result["prob"] == pytest.approx(1.0)

    def test_reversed_ordering_two_groups(self):
        y_true = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        y_pred = -y_true  # reversed
        groups = _make_groups(3, 3)
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert result["prob"] == pytest.approx(0.0)

    def test_single_member_groups_all_ignored(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        groups = np.array(["A", "B", "C"])  # each size 1
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert math.isnan(result["prob"])
        assert result["n_groups"] == 0
        assert result["n_pairs"] == 0

    def test_returns_required_keys(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1])
        groups = _make_groups(2, 2)
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert set(result.keys()) >= {"prob", "n_pairs", "n_groups"}

    def test_n_pairs_correct_for_two_groups_of_three(self):
        """
        Two groups of 3 non-tied elements → C(3,2)=3 pairs each → 6 total.
        """
        y_true = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        y_pred = np.array([1.1, 2.1, 3.1, 10.1, 20.1, 30.1])
        groups = _make_groups(3, 3)
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert result["n_pairs"] == 6

    def test_n_groups_excludes_singletons(self):
        # group A: 3 members, group B: 1 member → only A contributes
        y_true = np.array([1.0, 2.0, 3.0, 99.0])
        y_pred = np.array([1.1, 2.1, 3.1, 99.1])
        groups = np.array(["A", "A", "A", "B"])
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert result["n_groups"] == 1

    def test_between_group_pairs_ignored(self):
        """
        If we mix groups' orders the within-group result should still be perfect.
        Group A: [1,2,3] → [2,3,4] (correct order)
        Group B: [5,6,7] → [6,7,8] (correct order)
        Global order is not perfect but per-group it is.
        """
        y_true = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
        y_pred = np.array([2.0, 3.0, 4.0, 6.0, 7.0, 8.0])
        groups = _make_groups(3, 3)
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        assert result["prob"] == pytest.approx(1.0)

    def test_prob_range(self):
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 6, 60).astype(float)
        y_pred = rng.uniform(0, 5, 60)
        groups = rng.choice(["A", "B", "C", "D"], 60)
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        if not math.isnan(result["prob"]):
            assert 0.0 <= result["prob"] <= 1.0

    def test_ties_in_pred_excluded_from_pairs(self):
        """
        If all predicted values in a group are the same, no valid pairs exist
        for that group and it should not be counted.
        """
        # Group A: distinct true labels, all-tied pred → no valid pairs
        # Group B: normal data
        y_true = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        y_pred = np.array([2.5, 2.5, 2.5, 1.1, 2.1, 3.1])  # A tied, B fine
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        result = sr.within_group_pairwise_ordering_prob(y_true, y_pred, groups)
        # Only group B contributes → perfect ordering
        assert result["prob"] == pytest.approx(1.0)
        assert result["n_groups"] == 1


# ---------------------------------------------------------------------------
# within_group_spearman_correlation
# ---------------------------------------------------------------------------


class TestWithinGroupSpearmanCorrelation:
    """Size-weighted mean Spearman ρ per group."""

    def test_perfect_rank_order_gives_one(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        groups = _make_groups(4, 4)
        result = sr.within_group_spearman_correlation(y_true, y_true.copy(), groups)
        assert result["correlation"] == pytest.approx(1.0)

    def test_reversed_gives_minus_one(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        groups = _make_groups(4)
        result = sr.within_group_spearman_correlation(y_true, y_pred, groups)
        assert result["correlation"] == pytest.approx(-1.0)

    def test_single_member_groups_give_nan(self):
        y_true = np.array([1.0, 2.0, 3.0])
        groups = np.array(["A", "B", "C"])
        result = sr.within_group_spearman_correlation(y_true, y_true.copy(), groups)
        assert math.isnan(result["correlation"])
        assert result["n_groups"] == 0

    def test_returns_required_keys(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        groups = _make_groups(4)
        result = sr.within_group_spearman_correlation(y_true, y_true.copy(), groups)
        assert "correlation" in result
        assert "n_groups" in result

    def test_n_groups_counts_valid_groups(self):
        # Group A: 3 members, Group B: 1 member → only A valid
        y_true = np.array([1.0, 2.0, 3.0, 99.0])
        groups = np.array(["A", "A", "A", "B"])
        result = sr.within_group_spearman_correlation(y_true, y_true.copy(), groups)
        assert result["n_groups"] == 1

    def test_size_weighted_average(self):
        """
        Group A (4 obs): perfect order → ρ_A = 1.0
        Group B (2 obs): reversed → ρ_B = -1.0
        Weighted average: (1.0*4 + (-1.0)*2) / (4+2) = 2/6 ≈ 0.333
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 1.0])  # B reversed
        groups = np.array(["A", "A", "A", "A", "B", "B"])
        result = sr.within_group_spearman_correlation(y_true, y_pred, groups)
        expected = (1.0 * 4 + (-1.0) * 2) / 6.0
        assert result["correlation"] == pytest.approx(expected, abs=1e-9)

    def test_range_minus_one_to_one(self):
        rng = np.random.default_rng(7)
        y_true = rng.uniform(0, 5, 60)
        y_pred = rng.uniform(0, 5, 60)
        groups = rng.choice(["A", "B", "C"], 60)
        result = sr.within_group_spearman_correlation(y_true, y_pred, groups)
        if not math.isnan(result["correlation"]):
            assert -1.0 <= result["correlation"] <= 1.0


# ---------------------------------------------------------------------------
# org_year_pairwise_ordering_prob
# ---------------------------------------------------------------------------


class TestOrgYearPairwiseOrderingProb:
    """Within-org, within-year pairwise ordering probability."""

    @pytest.fixture
    def sample_data(self):
        """4 orgs × 3 years, 3 activities each."""
        rng = np.random.default_rng(11)
        n = 36
        orgs = np.repeat(["OrgA", "OrgB", "OrgC", "OrgD"], 9)
        years = np.tile(np.repeat(["2010", "2011", "2012"], 3), 4)
        y_true = rng.uniform(0, 5, n)
        y_pred = y_true + rng.normal(0, 0.1, n)  # near-perfect predictions
        return y_true, y_pred, orgs, years

    def test_returns_two_elements(self, sample_data):
        y_true, y_pred, orgs, years = sample_data
        result = sr.org_year_pairwise_ordering_prob(y_true, y_pred, orgs, years)
        assert len(result) == 2

    def test_weighted_avg_is_float_or_nan(self, sample_data):
        y_true, y_pred, orgs, years = sample_data
        weighted_avg, _ = sr.org_year_pairwise_ordering_prob(
            y_true, y_pred, orgs, years
        )
        assert isinstance(weighted_avg, float) or math.isnan(weighted_avg)

    def test_per_org_is_dict(self, sample_data):
        y_true, y_pred, orgs, years = sample_data
        _, per_org = sr.org_year_pairwise_ordering_prob(y_true, y_pred, orgs, years)
        assert isinstance(per_org, dict)

    def test_per_org_keys_match_unique_orgs(self, sample_data):
        y_true, y_pred, orgs, years = sample_data
        _, per_org = sr.org_year_pairwise_ordering_prob(y_true, y_pred, orgs, years)
        assert set(per_org.keys()) == {"OrgA", "OrgB", "OrgC", "OrgD"}

    def test_per_org_values_are_length_two_tuples(self, sample_data):
        y_true, y_pred, orgs, years = sample_data
        _, per_org = sr.org_year_pairwise_ordering_prob(y_true, y_pred, orgs, years)
        for org, val in per_org.items():
            assert len(val) == 2

    def test_per_org_n_pairs_non_negative(self, sample_data):
        y_true, y_pred, orgs, years = sample_data
        _, per_org = sr.org_year_pairwise_ordering_prob(y_true, y_pred, orgs, years)
        for org, (pop, n_pairs) in per_org.items():
            assert n_pairs >= 0

    def test_target_orgs_filters_output(self, sample_data):
        y_true, y_pred, orgs, years = sample_data
        _, per_org = sr.org_year_pairwise_ordering_prob(
            y_true, y_pred, orgs, years, target_orgs=["OrgA", "OrgB"]
        )
        assert set(per_org.keys()) == {"OrgA", "OrgB"}

    def test_near_perfect_predictions_give_high_prob(self, sample_data):
        """Near-perfect predictions should produce a high POP (>> 0.5)."""
        y_true, y_pred, orgs, years = sample_data
        weighted_avg, _ = sr.org_year_pairwise_ordering_prob(
            y_true, y_pred, orgs, years
        )
        if not math.isnan(weighted_avg):
            assert weighted_avg > 0.7

    def test_org_with_only_one_activity_gives_nan_pop(self):
        """An org that appears only once cannot form any within-year pair."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        orgs = np.array(["Solo", "Solo", "Solo"])
        years = np.array(["2010", "2011", "2012"])  # each year has 1 activity
        _, per_org = sr.org_year_pairwise_ordering_prob(y_true, y_pred, orgs, years)
        pop, n_pairs = per_org["Solo"]
        assert math.isnan(pop)
        assert n_pairs == 0

    def test_combined_group_key_consistent_with_within_group(self):
        """
        The weighted_avg from org_year_pairwise_ordering_prob should equal the
        result of calling within_group_pairwise_ordering_prob with org+year keys
        directly (that's exactly what the implementation does).
        """
        rng = np.random.default_rng(22)
        n = 30
        orgs = rng.choice(["A", "B"], n).astype(str)
        years = rng.choice(["2010", "2011", "2012"], n).astype(str)
        y_true = rng.uniform(0, 5, n)
        y_pred = rng.uniform(0, 5, n)

        weighted_avg, _ = sr.org_year_pairwise_ordering_prob(
            y_true, y_pred, orgs, years
        )

        group_keys = np.char.add(np.char.add(orgs, "|||"), years)
        ref = sr.within_group_pairwise_ordering_prob(y_true, y_pred, group_keys)

        if math.isnan(weighted_avg):
            assert math.isnan(ref["prob"])
        else:
            assert weighted_avg == pytest.approx(ref["prob"])
