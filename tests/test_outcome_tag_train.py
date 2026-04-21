"""
Pytest unit tests for pure functions and constants in:
  - src/pipeline/G_outcome_tag_train.py
  - src/pipeline/I_outcome_tag_shap_stability.py

Complements test_thesis_invariants.py (no duplication of already-tested items).
"""

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# G_outcome_tag_train — MANUAL_FACTORS structure
# ---------------------------------------------------------------------------


class TestManualFactors:
    """MANUAL_FACTORS defines three named factor groups with correct tag membership."""

    def test_three_factors_defined(self):
        from G_outcome_tag_train import MANUAL_FACTORS

        assert set(MANUAL_FACTORS.keys()) == {
            "factor_success",
            "factor_rescoping",
            "factor_finance",
        }

    def test_factor_success_positive_tags(self):
        from G_outcome_tag_train import MANUAL_FACTORS

        pos, neg = MANUAL_FACTORS["factor_success"]
        assert "tag_targets_met_or_exceeded_success" in pos
        assert "tag_capacity_building_delivered_success" in pos
        assert "tag_improved_financial_performance" in pos
        assert neg == []

    def test_factor_rescoping_positive_tags(self):
        from G_outcome_tag_train import MANUAL_FACTORS

        pos, neg = MANUAL_FACTORS["factor_rescoping"]
        assert "tag_project_restructured" in pos
        assert "tag_closing_date_extended" in pos
        assert "tag_funds_reallocated" in pos
        assert "tag_targets_revised" in pos
        assert neg == []

    def test_factor_finance_structure(self):
        from G_outcome_tag_train import MANUAL_FACTORS

        pos, neg = MANUAL_FACTORS["factor_finance"]
        assert "tag_high_disbursement" in pos
        assert "tag_funds_cancelled_or_unutilized" in neg

    def test_factors_are_mutually_exclusive(self):
        """No tag appears as a positive contributor in more than one factor."""
        from G_outcome_tag_train import MANUAL_FACTORS

        seen: set = set()
        for factor_name, (pos_tags, _) in MANUAL_FACTORS.items():
            for tag in pos_tags:
                assert (
                    tag not in seen
                ), f"Tag '{tag}' appears as positive in multiple factors"
                seen.add(tag)


# ---------------------------------------------------------------------------
# G_outcome_tag_train — TAGS_FACTOR_BLEND membership
# ---------------------------------------------------------------------------


class TestTagsFactorBlend:
    """TAGS_FACTOR_BLEND contains the expected success/finance tags."""

    def test_tags_factor_blend_is_nonempty(self):
        from G_outcome_tag_train import TAGS_FACTOR_BLEND

        assert len(TAGS_FACTOR_BLEND) > 0

    def test_high_disbursement_in_blend(self):
        from G_outcome_tag_train import TAGS_FACTOR_BLEND

        assert "tag_high_disbursement" in TAGS_FACTOR_BLEND

    def test_targets_met_in_blend(self):
        from G_outcome_tag_train import TAGS_FACTOR_BLEND

        assert "tag_targets_met_or_exceeded_success" in TAGS_FACTOR_BLEND

    def test_closing_date_not_in_blend(self):
        """Rescoping tags are NOT factor-blended (rescoping group disbanded)."""
        from G_outcome_tag_train import TAGS_FACTOR_BLEND

        assert "tag_closing_date_extended" not in TAGS_FACTOR_BLEND

    def test_project_restructured_not_in_blend(self):
        from G_outcome_tag_train import TAGS_FACTOR_BLEND

        assert "tag_project_restructured" not in TAGS_FACTOR_BLEND


# ---------------------------------------------------------------------------
# G_outcome_tag_train — _compute_manual_factor_scores (pure function)
# ---------------------------------------------------------------------------


class TestComputeManualFactorScores:
    """_compute_manual_factor_scores computes factor = mean(pos) - mean(neg)."""

    def _make_tag_df(self, n=10):
        import pandas as pd

        rng = np.random.default_rng(0)
        cols = [
            "tag_targets_met_or_exceeded_success",
            "tag_high_beneficiary_satisfaction_or_reach_success",
            "tag_private_sector_engagement_success",
            "tag_capacity_building_delivered_success",
            "tag_policy_regulatory_reforms_success_success",
            "tag_improved_financial_performance",
            "tag_project_restructured",
            "tag_closing_date_extended",
            "tag_funds_reallocated",
            "tag_targets_revised",
            "tag_high_disbursement",
            "tag_funds_cancelled_or_unutilized",
        ]
        data = {c: rng.integers(0, 2, n).astype(float) for c in cols}
        return pd.DataFrame(data)

    def test_returns_three_factor_columns(self):
        from G_outcome_tag_train import _compute_manual_factor_scores

        df = self._make_tag_df()
        result = _compute_manual_factor_scores(df)
        assert set(result.columns) == {
            "factor_success",
            "factor_rescoping",
            "factor_finance",
        }

    def test_index_preserved(self):
        import pandas as pd
        from G_outcome_tag_train import _compute_manual_factor_scores

        df = self._make_tag_df(n=8)
        result = _compute_manual_factor_scores(df)
        assert list(result.index) == list(df.index)

    def test_factor_finance_all_one_disbursement_no_cancelled(self):
        """factor_finance = mean(pos) - mean(neg) = 1.0 - 0.0 = 1.0 when disbursement=1, cancelled=0."""
        import pandas as pd
        from G_outcome_tag_train import _compute_manual_factor_scores

        n = 5
        df = pd.DataFrame(
            {
                "tag_high_disbursement": np.ones(n),
                "tag_funds_cancelled_or_unutilized": np.zeros(n),
            }
        )
        result = _compute_manual_factor_scores(df)
        assert np.allclose(result["factor_finance"].values, 1.0)

    def test_factor_finance_all_zero_disbursement_all_one_cancelled(self):
        """factor_finance = 0.0 - 1.0 = -1.0."""
        import pandas as pd
        from G_outcome_tag_train import _compute_manual_factor_scores

        n = 4
        df = pd.DataFrame(
            {
                "tag_high_disbursement": np.zeros(n),
                "tag_funds_cancelled_or_unutilized": np.ones(n),
            }
        )
        result = _compute_manual_factor_scores(df)
        assert np.allclose(result["factor_finance"].values, -1.0)

    def test_factor_success_range(self):
        """factor_success is always in [-1, 1] since it is a mean of binary values."""
        from G_outcome_tag_train import _compute_manual_factor_scores

        df = self._make_tag_df(n=50)
        result = _compute_manual_factor_scores(df)
        assert result["factor_success"].between(-1.0, 1.0).all()

    def test_empty_pos_neg_yields_zero(self):
        """If the DataFrame has none of the factor's tags, score should be 0.0."""
        import pandas as pd
        from G_outcome_tag_train import _compute_manual_factor_scores

        df = pd.DataFrame({"unrelated_col": [1, 0, 1]})
        result = _compute_manual_factor_scores(df)
        # All factors whose tags are absent -> pos_mean = 0, neg_mean = 0 -> 0
        assert (result == 0.0).all().all()


# ---------------------------------------------------------------------------
# I_outcome_tag_shap_stability — get_target_tags
# ---------------------------------------------------------------------------


class TestGetTargetTags:
    """get_target_tags() must return exactly the same set as HARDCODED_14_TAGS."""

    def test_get_target_tags_matches_hardcoded_14(self):
        from I_outcome_tag_shap_stability import get_target_tags
        from G_outcome_tag_train import HARDCODED_14_TAGS

        assert set(get_target_tags()) == set(HARDCODED_14_TAGS)

    def test_get_target_tags_returns_list(self):
        from I_outcome_tag_shap_stability import get_target_tags

        result = get_target_tags()
        assert isinstance(result, list)

    def test_get_target_tags_no_duplicates(self):
        from I_outcome_tag_shap_stability import get_target_tags

        result = get_target_tags()
        assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# I_outcome_tag_shap_stability — get_active_feat_cols
# ---------------------------------------------------------------------------


class TestGetActiveFeatCols:
    """get_active_feat_cols behaviour for all four documented cases."""

    @pytest.fixture
    def feature_cols(self):
        return ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]

    def test_feat_idx_none_returns_all_feature_cols(self, feature_cols):
        """When strat has feat_idx=None → returns all feature_cols unchanged."""
        from I_outcome_tag_shap_stability import get_active_feat_cols

        per_tag_strats = {"tag_x": {"strategy": "top5", "feat_idx": None}}
        result = get_active_feat_cols("tag_x", per_tag_strats, feature_cols)
        assert result == feature_cols

    def test_feat_idx_list_returns_selected_cols(self, feature_cols):
        """When feat_idx=[0, 2] → returns feature_cols[0] and feature_cols[2]."""
        from I_outcome_tag_shap_stability import get_active_feat_cols

        per_tag_strats = {"tag_x": {"strategy": "top5", "feat_idx": [0, 2]}}
        result = get_active_feat_cols("tag_x", per_tag_strats, feature_cols)
        assert result == ["feat_a", "feat_c"]

    def test_feat_idx_out_of_range_falls_back(self, feature_cols):
        """feat_idx values all beyond len(feature_cols) → empty → falls back to feature_cols."""
        from I_outcome_tag_shap_stability import get_active_feat_cols

        per_tag_strats = {"tag_x": {"strategy": "top5", "feat_idx": [100, 200]}}
        result = get_active_feat_cols("tag_x", per_tag_strats, feature_cols)
        assert result == feature_cols

    def test_unknown_tag_falls_back_to_all(self, feature_cols):
        """Tag absent from per_tag_strats → defaults to feat_idx=None → all feature_cols."""
        from I_outcome_tag_shap_stability import get_active_feat_cols

        per_tag_strats = {"tag_other": {"strategy": "top5", "feat_idx": [1]}}
        result = get_active_feat_cols("tag_unknown", per_tag_strats, feature_cols)
        assert result == feature_cols

    @pytest.mark.parametrize(
        "feat_idx,expected_names",
        [
            ([0], ["feat_a"]),
            ([1, 3], ["feat_b", "feat_d"]),
            ([0, 1, 2, 3, 4], ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]),
            ([4, 0], ["feat_e", "feat_a"]),
        ],
    )
    def test_various_feat_idx_selections(self, feature_cols, feat_idx, expected_names):
        from I_outcome_tag_shap_stability import get_active_feat_cols

        per_tag_strats = {"tag_t": {"feat_idx": feat_idx}}
        result = get_active_feat_cols("tag_t", per_tag_strats, feature_cols)
        assert result == expected_names

    def test_partial_out_of_range_keeps_valid(self, feature_cols):
        """Mixed in-range and out-of-range indices → only valid columns returned (not fallback)."""
        from I_outcome_tag_shap_stability import get_active_feat_cols

        per_tag_strats = {"tag_x": {"feat_idx": [0, 999]}}
        result = get_active_feat_cols("tag_x", per_tag_strats, feature_cols)
        # 0 is valid → returns ["feat_a"]; since non-empty, no fallback
        assert result == ["feat_a"]

    def test_empty_feature_cols_fallback(self):
        """If feature_cols itself is empty, fallback also returns empty list."""
        from I_outcome_tag_shap_stability import get_active_feat_cols

        result = get_active_feat_cols("tag_x", {}, [])
        assert result == []


# ---------------------------------------------------------------------------
# I_outcome_tag_shap_stability — split_stability_stats (pure computation)
# ---------------------------------------------------------------------------


class TestSplitStabilityStats:
    """split_stability_stats computes overlap, sign-flip rate, and rank correlation."""

    def _make_uniform_splits(self, n_feats=5, n_splits=3):
        """All splits have identical SHAP importances -> perfect rank correlation, no sign flips."""
        base = np.arange(n_feats, dtype=float)
        abs_means = [base.copy() for _ in range(n_splits)]
        signed_means = [base.copy() for _ in range(n_splits)]
        return abs_means, signed_means

    def test_returns_required_keys(self):
        from I_outcome_tag_shap_stability import split_stability_stats

        feat_cols = [f"f{i}" for i in range(5)]
        abs_m, sign_m = self._make_uniform_splits(n_feats=5)
        result = split_stability_stats(feat_cols, abs_m, sign_m, top_k=3)
        assert {
            "k",
            "in_all",
            "avg_pairwise_overlap",
            "sign_flips",
            "avg_rank_corr",
        } <= set(result)

    def test_identical_splits_perfect_rank_corr(self):
        """When all splits have the same importances, rank correlation should be 1.0."""
        from I_outcome_tag_shap_stability import split_stability_stats

        feat_cols = [f"f{i}" for i in range(6)]
        abs_m, sign_m = self._make_uniform_splits(n_feats=6, n_splits=3)
        result = split_stability_stats(feat_cols, abs_m, sign_m, top_k=3)
        assert result["avg_rank_corr"] == pytest.approx(1.0)

    def test_identical_splits_no_sign_flips(self):
        """When all splits agree on signs, sign_flips should be 0."""
        from I_outcome_tag_shap_stability import split_stability_stats

        feat_cols = [f"f{i}" for i in range(5)]
        abs_m, sign_m = self._make_uniform_splits(n_feats=5)
        result = split_stability_stats(feat_cols, abs_m, sign_m, top_k=3)
        assert result["sign_flips"] == 0

    def test_identical_splits_full_topk_overlap(self):
        """When all splits are identical, all top-k features appear in every split."""
        from I_outcome_tag_shap_stability import split_stability_stats

        feat_cols = [f"f{i}" for i in range(5)]
        abs_m, sign_m = self._make_uniform_splits(n_feats=5)
        result = split_stability_stats(feat_cols, abs_m, sign_m, top_k=3)
        assert result["in_all"] == result["k"]

    def test_k_capped_by_feature_count(self):
        """top_k > len(feat_cols) should be capped to len(feat_cols)."""
        from I_outcome_tag_shap_stability import split_stability_stats

        feat_cols = ["a", "b"]
        abs_m = [np.array([1.0, 2.0]), np.array([2.0, 1.0])]
        sign_m = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        result = split_stability_stats(feat_cols, abs_m, sign_m, top_k=10)
        assert result["k"] == 2

    def test_sign_flip_detected(self):
        """A feature that is positive in one split and negative in another is a sign flip."""
        from I_outcome_tag_shap_stability import split_stability_stats

        feat_cols = ["a", "b", "c"]
        # abs importances: feature 'a' (index 2) is always most important
        abs_m = [
            np.array([0.1, 0.2, 1.0]),
            np.array([0.1, 0.2, 1.0]),
        ]
        # signed: feature 'a' flips sign between splits
        sign_m = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, -1.0]),
        ]
        result = split_stability_stats(feat_cols, abs_m, sign_m, top_k=1)
        assert result["sign_flips"] >= 1
