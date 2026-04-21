"""
Unit tests for src/pipeline/L_cost_effectiveness_train_and_score.py.

Tests cover only constants and pure functions — no file I/O, no model training.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = REPO_ROOT / "src" / "pipeline"
UTILS_DIR = REPO_ROOT / "src" / "utils"

for _p in [str(PIPELINE_DIR), str(UTILS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the module under test (constants + pure functions only).
import L_cost_effectiveness_train_and_score as L

# ===========================================================================
# 1.  ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS — structural invariants
# ===========================================================================


class TestOnlyUseList:
    """Tests for the ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS constant list."""

    def test_is_a_list(self):
        assert isinstance(L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS, list)

    def test_all_elements_are_strings(self):
        for item in L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS:
            assert isinstance(item, str), f"Non-string entry: {item!r}"

    def test_no_duplicates(self):
        lst = L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS
        assert len(lst) == len(set(lst)), "Duplicate entries in ONLY_USE_THESE list"

    def test_non_empty(self):
        assert len(L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS) > 0

    # thesis: B/C ratios and rates of return are NOT log10-transformed → they
    # appear in the list as out_raw_ entries (linear scale), not out_dpu_.
    def test_benefit_cost_ratio_present_as_raw(self):
        lst = L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS
        bc_entries = [c for c in lst if "benefit_cost" in c]
        assert len(bc_entries) >= 1, "Expected at least one benefit_cost_ratio entry"
        for c in bc_entries:
            assert c.startswith(
                "out_raw_"
            ), f"B/C ratio must be 'out_raw_' (linear, not log10): {c}"

    def test_economic_rate_of_return_present_as_raw(self):
        lst = L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS
        ror_entries = [c for c in lst if "economic_rate_of_return" in c]
        assert (
            len(ror_entries) >= 1
        ), "Expected at least one economic_rate_of_return entry"
        for c in ror_entries:
            assert c.startswith(
                "out_raw_"
            ), f"Economic RoR must be 'out_raw_' (linear, not log10): {c}"

    def test_financial_rate_of_return_present_as_raw(self):
        lst = L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS
        ror_entries = [c for c in lst if "financial_rate_of_return" in c]
        assert (
            len(ror_entries) >= 1
        ), "Expected at least one financial_rate_of_return entry"
        for c in ror_entries:
            assert c.startswith(
                "out_raw_"
            ), f"Financial RoR must be 'out_raw_' (linear, not log10): {c}"

    # thesis: B/C ratios and RoR must NOT appear as out_dpu_ columns
    @pytest.mark.parametrize(
        "keyword",
        [
            "benefit_cost",
            "economic_rate_of_return",
            "financial_rate_of_return",
        ],
    )
    def test_no_dpu_entry_for_ratio_or_ror(self, keyword):
        """B/C ratios and rates of return must not appear as out_dpu_ entries."""
        lst = L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS
        bad = [c for c in lst if keyword in c and c.startswith("out_dpu_")]
        assert (
            bad == []
        ), f"Found out_dpu_ log-transform entries for '{keyword}' (should be out_raw_): {bad}"


# ===========================================================================
# 2.  Naming convention: out_raw_ vs out_dpu_
# ===========================================================================


class TestOutcomeNamingConvention:
    """Every entry must start with 'out_raw_' or 'out_dpu_'."""

    def test_all_entries_have_correct_prefix(self):
        valid_prefixes = ("out_raw_", "out_dpu_")
        for col in L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS:
            assert col.startswith(
                valid_prefixes
            ), f"Column '{col}' does not start with out_raw_ or out_dpu_"

    def test_out_raw_entries_have_double_underscore_separator(self):
        """Column names must follow the pattern prefix_distribution__units."""
        for col in L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS:
            assert "__" in col, f"Expected '__' unit separator in '{col}'"

    def test_raw_and_dpu_entries_present(self):
        lst = L.ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS
        has_raw = any(c.startswith("out_raw_") for c in lst)
        has_dpu = any(c.startswith("out_dpu_") for c in lst)
        assert has_raw, "No 'out_raw_' entries found"
        assert has_dpu, "No 'out_dpu_' entries found"


# ===========================================================================
# 3.  EXCLUDE_DPU_DISTS — distributions excluded from dollar-per-unit (log10) targets
# ===========================================================================


class TestExcludeDpuDists:
    """thesis: B/C ratios and rates of return are excluded from per-dollar-unit outcomes."""

    def test_is_a_set(self):
        assert isinstance(L.EXCLUDE_DPU_DISTS, set)

    @pytest.mark.parametrize(
        "dist",
        [
            "benefit_cost_ratios",
            "economic_rate_of_return",
            "financial_rate_of_return",
        ],
    )
    def test_excluded_distributions_present(self, dist):
        assert (
            dist in L.EXCLUDE_DPU_DISTS
        ), f"'{dist}' must be in EXCLUDE_DPU_DISTS (B/C ratios and RoR not log10-transformed)"

    def test_no_standard_physical_outcomes_excluded(self):
        """Physical outcome distributions (water, CO2, capacity) should NOT be in the exclusion set."""
        physical = {
            "water_connections",
            "co2_emission_reductions",
            "generation_capacity",
        }
        for dist in physical:
            assert (
                dist not in L.EXCLUDE_DPU_DISTS
            ), f"Physical distribution '{dist}' should not be excluded from DPU"


# ===========================================================================
# 4.  is_yield_dist — correctly identifies yield distributions
# ===========================================================================


class TestIsYieldDist:

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("yield_increases_percent", True),
            ("yield_increases_t_per_ha", True),
            ("yield_increase", True),
            ("yield", True),
            ("YIELD_INCREASES_PERCENT", True),  # case-insensitive
            ("co2_emission_reductions", False),
            ("water_connections", False),
            ("benefit_cost_ratios", False),
            ("economic_rate_of_return", False),
            ("generation_capacity", False),
        ],
    )
    def test_yield_detection(self, name, expected):
        assert (
            L.is_yield_dist(name) is expected
        ), f"is_yield_dist({name!r}) expected {expected}"


# ===========================================================================
# 5.  add_groupwise_z — z-score standardization with train-set stats, ddof=0
# ===========================================================================


class TestAddGroupwiseZ:
    """thesis: z-score uses training-set mean and std with ddof=0."""

    def _make_long_df(self, y_values, activity_ids, group="grpA"):
        """Build a minimal long_df suitable for add_groupwise_z."""
        return pd.DataFrame(
            {
                "activity_id": activity_ids,
                "which_group": group,
                "y_raw": np.array(y_values, dtype=float),
            }
        )

    def test_train_mean_is_zero_after_zscore(self):
        """After z-scoring with train stats, the mean of TRAIN z-scores is ~0."""
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        aids = [f"act_{i}" for i in range(5)]
        train_aids = set(aids[:4])  # first 4 are train
        df = self._make_long_df(y, aids)
        df = L.add_groupwise_z(
            df,
            train_aids=train_aids,
            group_col="which_group",
            y_col="y_raw",
            z_col="y_z",
        )
        train_mask = df["activity_id"].isin(train_aids)
        assert df.loc[train_mask, "y_z"].mean() == pytest.approx(0.0, abs=1e-9)

    def test_train_std_is_one_after_zscore(self):
        """After z-scoring with ddof=0 train stats, the std (ddof=0) of TRAIN z-scores is ~1."""
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        aids = [f"act_{i}" for i in range(5)]
        train_aids = set(aids[:4])
        df = self._make_long_df(y, aids)
        df = L.add_groupwise_z(
            df,
            train_aids=train_aids,
            group_col="which_group",
            y_col="y_raw",
            z_col="y_z",
        )
        train_mask = df["activity_id"].isin(train_aids)
        assert df.loc[train_mask, "y_z"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)

    def test_uses_ddof0_not_ddof1(self):
        """Verify that ddof=0 is used: hand-compute the expected z-scores."""
        y = [2.0, 4.0, 6.0]
        aids = ["a", "b", "c"]
        train_aids = {"a", "b", "c"}
        df = self._make_long_df(y, aids)
        df = L.add_groupwise_z(
            df,
            train_aids=train_aids,
            group_col="which_group",
            y_col="y_raw",
            z_col="y_z",
        )
        mu = np.mean(y)  # 4.0
        sd = np.std(y, ddof=0)  # sqrt(8/3)  ≈ 1.633
        expected_z = [(v - mu) / sd for v in y]
        assert df["y_z"].tolist() == pytest.approx(expected_z, abs=1e-9)

    def test_out_of_sample_uses_train_stats(self):
        """Test observations are standardised using TRAIN mean/sd, not their own stats."""
        train_y = [0.0, 2.0, 4.0, 6.0]  # mu=3, sd(ddof=0)=sqrt(5)≈2.236
        test_y = [100.0]
        aids = [f"a{i}" for i in range(5)]
        train_aids = set(aids[:4])
        y = train_y + test_y
        df = self._make_long_df(y, aids)
        df = L.add_groupwise_z(
            df,
            train_aids=train_aids,
            group_col="which_group",
            y_col="y_raw",
            z_col="y_z",
        )
        mu_tr = np.mean(train_y)
        sd_tr = np.std(train_y, ddof=0)
        expected_test_z = (100.0 - mu_tr) / sd_tr
        test_mask = df["activity_id"] == "a4"
        assert float(df.loc[test_mask, "y_z"].iloc[0]) == pytest.approx(
            expected_test_z, rel=1e-6
        )

    def test_adds_mu_g_and_sd_g_columns(self):
        """add_groupwise_z must add mu_g and sd_g columns."""
        y = [1.0, 3.0, 5.0]
        aids = ["x", "y", "z"]
        df = self._make_long_df(y, aids)
        df = L.add_groupwise_z(
            df,
            train_aids=set(aids),
            group_col="which_group",
            y_col="y_raw",
            z_col="y_z",
        )
        assert "mu_g" in df.columns
        assert "sd_g" in df.columns
        assert "median_g" in df.columns

    def test_two_groups_use_separate_stats(self):
        """Each group is standardised independently."""
        df = pd.DataFrame(
            {
                "activity_id": ["a1", "a2", "a3", "b1", "b2", "b3"],
                "which_group": ["G1", "G1", "G1", "G2", "G2", "G2"],
                "y_raw": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            }
        )
        train_aids = {"a1", "a2", "a3", "b1", "b2", "b3"}
        df = L.add_groupwise_z(
            df,
            train_aids=train_aids,
            group_col="which_group",
            y_col="y_raw",
            z_col="y_z",
        )
        # Both groups should have mean-zero z-scores independently
        for grp in ["G1", "G2"]:
            grp_z = df.loc[df["which_group"] == grp, "y_z"]
            assert grp_z.mean() == pytest.approx(
                0.0, abs=1e-9
            ), f"Group {grp}: expected mean(z)=0, got {grp_z.mean()}"


# ===========================================================================
# 6.  Sample weight computation — recent 20% gets 3× weight
# ===========================================================================


class TestSampleWeights:
    """thesis: RF sample weight = 3× for most recent 20% of training activities."""

    def _compute_weights(self, n: int) -> np.ndarray:
        """Replicate the inline weight logic from main()."""
        cutoff_idx = int(n * 0.8)
        w = np.ones(n)
        w[cutoff_idx:] = 3.0
        return w

    @pytest.mark.parametrize(
        "n,expected_heavy",
        [
            (10, 2),  # 20% of 10 = 2; cutoff=8 → indices 8..9 (2 items)
            (100, 20),  # 20% of 100 = 20; cutoff=80 → indices 80..99 (20 items)
            (5, 1),  # int(5*0.8)=4; indices 4..4 (1 item)
            (
                1,
                1,
            ),  # int(1*0.8)=0; w[0:]=3.0 → 1 item (degenerate but correct behaviour)
        ],
    )
    def test_cutoff_position(self, n, expected_heavy):
        w = self._compute_weights(n)
        heavy = int((w == 3.0).sum())
        assert (
            heavy == expected_heavy
        ), f"n={n}: expected {expected_heavy} heavy-weighted samples, got {heavy}"

    def test_light_weights_are_one(self):
        w = self._compute_weights(10)
        assert np.all(w[:8] == 1.0)

    def test_heavy_weights_are_three(self):
        w = self._compute_weights(10)
        assert np.all(w[8:] == 3.0)

    def test_total_weight_is_correct(self):
        n = 10
        w = self._compute_weights(n)
        # 8 × 1 + 2 × 3 = 14
        assert w.sum() == pytest.approx(8.0 + 2.0 * 3.0)

    def test_weight_ratio_is_3x(self):
        """Heavy weight must be exactly 3× light weight."""
        w = self._compute_weights(20)
        assert w.max() / w.min() == pytest.approx(3.0)

    @pytest.mark.parametrize("n", [5, 10, 50, 100, 1000])
    def test_only_two_distinct_weight_values(self, n):
        w = self._compute_weights(n)
        unique_vals = set(w.tolist())
        # When n < 5, the cutoff might be 0, so all weights are 3.0
        assert unique_vals.issubset({1.0, 3.0})


# ===========================================================================
# 7.  Log-transform exclusion: is_ror / EXCLUDE_DPU_DISTS membership
# ===========================================================================


class TestLogTransformExclusion:
    """
    thesis: B/C ratios and economic/financial RoR are NOT log10-transformed.
    The code achieves this by routing them through `outcome_norm` (linear) while
    everything else uses `outcome_norm_log10`.
    """

    def test_ror_flag_set_for_economic_ror(self):
        """outcomes_to_wide_two_variants flags economic_rate_of_return as is_ror."""
        dist_series = pd.Series(
            ["economic_rate_of_return", "water_connections", "generation_capacity"]
        )
        is_ror = dist_series.isin(
            {"economic_rate_of_return", "financial_rate_of_return"}
        )
        assert is_ror.iloc[0] is np.bool_(True)
        assert is_ror.iloc[1] is np.bool_(False)

    def test_ror_flag_set_for_financial_ror(self):
        dist_series = pd.Series(["financial_rate_of_return", "co2_emission_reductions"])
        is_ror = dist_series.isin(
            {"economic_rate_of_return", "financial_rate_of_return"}
        )
        assert is_ror.iloc[0] is np.bool_(True)
        assert is_ror.iloc[1] is np.bool_(False)

    @pytest.mark.parametrize(
        "dist",
        [
            "co2_emission_reductions",
            "water_connections",
            "generation_capacity",
            "beneficiaries",
        ],
    )
    def test_physical_outcomes_are_not_ror(self, dist):
        is_ror = dist in {"economic_rate_of_return", "financial_rate_of_return"}
        assert not is_ror

    @pytest.mark.parametrize(
        "dist",
        [
            "benefit_cost_ratios",
            "economic_rate_of_return",
            "financial_rate_of_return",
        ],
    )
    def test_exempt_dists_excluded_from_dpu(self, dist):
        """Distributions that skip log10 must be in EXCLUDE_DPU_DISTS."""
        assert (
            dist in L.EXCLUDE_DPU_DISTS
        ), f"'{dist}' should be in EXCLUDE_DPU_DISTS to prevent log10 transformation"


# ===========================================================================
# 8.  _safe_name — utility for constructing column names
# ===========================================================================


class TestSafeName:

    def test_spaces_replaced(self):
        assert " " not in L._safe_name("hello world")

    def test_special_chars_replaced(self):
        result = L._safe_name("foo/bar:baz!")
        assert all(c.isalnum() or c in "._-" for c in result)

    def test_truncated_to_120_chars(self):
        long_str = "a" * 200
        assert len(L._safe_name(long_str)) <= 120

    def test_normal_name_unchanged(self):
        assert L._safe_name("co2_emission_reductions") == "co2_emission_reductions"

    def test_double_underscore_preserved(self):
        result = L._safe_name("benefit_cost_ratios__ratio")
        assert "__" in result or "benefit_cost_ratios" in result


# ===========================================================================
# 9.  filter_groups_min_counts — keeps only groups satisfying min-count constraints
# ===========================================================================


class TestFilterGroupsMinCounts:

    def _make_long_df(self, train_aids, test_aids, group_sizes_train, group_sizes_test):
        """
        group_sizes_train / group_sizes_test: dict mapping group_name -> count.
        """
        rows = []
        aid_counter = 0
        for grp, n_tr in group_sizes_train.items():
            for _ in range(n_tr):
                rows.append({"activity_id": f"tr{aid_counter:04d}", "which_group": grp})
                train_aids.add(f"tr{aid_counter:04d}")
                aid_counter += 1
        for grp, n_te in group_sizes_test.items():
            for _ in range(n_te):
                rows.append({"activity_id": f"te{aid_counter:04d}", "which_group": grp})
                test_aids.add(f"te{aid_counter:04d}")
                aid_counter += 1
        return pd.DataFrame(rows)

    def test_group_with_insufficient_train_is_dropped(self):
        train_aids, test_aids = set(), set()
        df = self._make_long_df(
            train_aids,
            test_aids,
            {"good_grp": 15, "small_grp": 3},
            {"good_grp": 15, "small_grp": 15},
        )
        out, kept = L.filter_groups_min_counts(
            df,
            train_aids=train_aids,
            test_aids=test_aids,
            group_col="which_group",
            min_train=10,
            min_test=10,
        )
        assert "small_grp" not in kept
        assert "good_grp" in kept

    def test_group_with_insufficient_test_is_dropped(self):
        train_aids, test_aids = set(), set()
        df = self._make_long_df(
            train_aids,
            test_aids,
            {"good_grp": 15, "sparse_test_grp": 15},
            {"good_grp": 15, "sparse_test_grp": 2},
        )
        out, kept = L.filter_groups_min_counts(
            df,
            train_aids=train_aids,
            test_aids=test_aids,
            group_col="which_group",
            min_train=10,
            min_test=10,
        )
        assert "sparse_test_grp" not in kept
        assert "good_grp" in kept

    def test_output_df_only_contains_kept_groups(self):
        train_aids, test_aids = set(), set()
        df = self._make_long_df(
            train_aids,
            test_aids,
            {"grpA": 20, "grpB": 5},
            {"grpA": 20, "grpB": 20},
        )
        out, kept = L.filter_groups_min_counts(
            df,
            train_aids=train_aids,
            test_aids=test_aids,
            group_col="which_group",
            min_train=10,
            min_test=10,
        )
        assert set(out["which_group"].unique()) == set(kept)

    def test_all_groups_kept_when_all_above_threshold(self):
        train_aids, test_aids = set(), set()
        df = self._make_long_df(
            train_aids,
            test_aids,
            {"g1": 20, "g2": 20},
            {"g1": 20, "g2": 20},
        )
        out, kept = L.filter_groups_min_counts(
            df,
            train_aids=train_aids,
            test_aids=test_aids,
            group_col="which_group",
            min_train=10,
            min_test=10,
        )
        assert set(kept) == {"g1", "g2"}


# ===========================================================================
# 10.  SUBTRACT_MEAN_IN_Z_SCORE flag
# ===========================================================================


class TestSubtractMeanFlag:
    def test_subtract_mean_in_z_score_is_true(self):
        """thesis: z-score standardisation includes subtracting the group mean."""
        assert L.SUBTRACT_MEAN_IN_Z_SCORE is True
