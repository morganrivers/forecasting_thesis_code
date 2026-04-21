"""
Unit tests for:
  - J_outcome_tag_results_table.py  (FEAT_ABBREV dict, SHAP consistency constants)
  - K_outcome_tag_extrapolate_scaling.py  (learning-curve / sliding-window constants)
  - M_data_analysis_eval_set_sizes.py  (SOURCES dict, load_split function)
"""

from __future__ import annotations

import sys
from pathlib import Path

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


# ===========================================================================
# J — outcome tag results table
# ===========================================================================


class TestJAbbreviations:
    """Tests for FEAT_ABBREV dict in J_outcome_tag_results_table.py."""

    @pytest.fixture(scope="class")
    def feat_abbrev(self):
        from J_outcome_tag_results_table import FEAT_ABBREV

        return FEAT_ABBREV

    def test_dict_is_non_empty(self, feat_abbrev):
        assert len(feat_abbrev) > 0

    @pytest.mark.parametrize("key", ["umap3_x", "umap3_y", "umap3_z"])
    def test_umap_keys_map_to_U_label(self, feat_abbrev, key):
        val = feat_abbrev[key]
        assert (
            "U" in val or "umap" in val.lower()
        ), f"UMAP key '{key}' should map to something containing 'U' or 'umap', got '{val}'"

    @pytest.mark.parametrize(
        "key,expected",
        [
            # raw expenditure and the collapsed group both abbreviate to "exp"
            ("planned_expenditure", "exp"),
            ("expenditure_grp", "exp"),
            # the log transforms have their own short abbreviations
            ("log_planned_expenditure", "lexp"),
            ("expenditure_per_year_log", "eyr"),
        ],
    )
    def test_expenditure_features_abbreviation(self, feat_abbrev, key, expected):
        assert (
            feat_abbrev[key] == expected
        ), f"Expenditure key '{key}' should map to '{expected}', got '{feat_abbrev[key]}'"

    def test_duration_maps_to_dur(self, feat_abbrev):
        assert feat_abbrev["planned_duration"] == "dur"

    def test_scope_maps_to_scp(self, feat_abbrev):
        assert feat_abbrev["activity_scope"] == "scp"

    def test_loan_indicator_maps_to_loan(self, feat_abbrev):
        assert feat_abbrev["finance_is_loan"] == "loan"


class TestJSHAPConstants:
    """Tests for SHAP-consistency filter constants embedded in get_consistent_features."""

    def test_get_consistent_features_importable(self):
        from J_outcome_tag_results_table import get_consistent_features

        assert callable(get_consistent_features)

    def test_default_top_k_is_10(self):
        """Docstring states 'top 10 by |SHAP|'; default top_k parameter must be 10."""
        import inspect
        from J_outcome_tag_results_table import get_consistent_features

        sig = inspect.signature(get_consistent_features)
        assert sig.parameters["top_k"].default == 10

    def test_default_min_importance_frac_is_0_05(self):
        """Docstring states 'at least 5% of total |SHAP|'; default min_importance_frac must be 0.05."""
        import inspect
        from J_outcome_tag_results_table import get_consistent_features

        sig = inspect.signature(get_consistent_features)
        assert sig.parameters["min_importance_frac"].default == pytest.approx(0.05)

    def test_same_sign_filter_excludes_sign_flip(self):
        """A feature whose sign flips between splits must be excluded."""
        import numpy as np
        from J_outcome_tag_results_table import get_consistent_features

        # Two features; feature 0 has consistent sign, feature 1 flips
        feature_cols = ["feat_stable", "feat_flip"]
        split_abs_means = [
            [0.5, 0.3],
            [0.5, 0.3],
        ]
        split_signed_means = [
            [0.5, 0.3],  # split 0: both positive
            [0.5, -0.3],  # split 1: feat_flip goes negative
        ]
        result = get_consistent_features(
            feature_cols,
            split_abs_means,
            split_signed_means,
            top_k=10,
            min_importance_frac=0.0,
        )
        result_feats = [r[0] for r in result]
        assert "feat_stable" in result_feats
        assert "feat_flip" not in result_feats

    def test_top_k_filter_excludes_low_rank_feature(self):
        """A feature outside the top-k in any split must be excluded."""
        import numpy as np
        from J_outcome_tag_results_table import get_consistent_features

        n = 12
        feature_cols = [f"f{i}" for i in range(n)]
        # f0 has high importance in split 0 but drops to rank 11 in split 1
        split_abs_means = [
            [1.0] + [0.1] * (n - 1),  # split 0: f0 is #1
            [0.0] + [0.1] * (n - 1),  # split 1: f0 is last
        ]
        split_signed_means = [
            [1.0] + [0.1] * (n - 1),
            [0.0] + [0.1] * (n - 1),
        ]
        result = get_consistent_features(
            feature_cols,
            split_abs_means,
            split_signed_means,
            top_k=5,
            min_importance_frac=0.0,
        )
        result_feats = [r[0] for r in result]
        assert "f0" not in result_feats


# ===========================================================================
# K — outcome tag extrapolate scaling
# ===========================================================================


class TestKConstants:
    """Tests for module-level constants in K_outcome_tag_extrapolate_scaling.py."""

    @pytest.fixture(scope="class")
    def K(self):
        import K_outcome_tag_extrapolate_scaling as _K

        return _K

    def test_n_lc_drop_fracs_length(self, K):
        assert len(K.N_LC_DROP_FRACS) == 10

    def test_n_lc_drop_fracs_starts_at_zero(self, K):
        assert K.N_LC_DROP_FRACS[0] == pytest.approx(0.0)

    def test_n_lc_drop_fracs_ends_at_0_9(self, K):
        assert K.N_LC_DROP_FRACS[-1] == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "i, expected",
        [
            (0, 0.0),
            (1, 0.1),
            (2, 0.2),
            (3, 0.3),
            (4, 0.4),
            (5, 0.5),
            (6, 0.6),
            (7, 0.7),
            (8, 0.8),
            (9, 0.9),
        ],
    )
    def test_n_lc_drop_fracs_values(self, K, i, expected):
        assert K.N_LC_DROP_FRACS[i] == pytest.approx(expected)

    def test_n_lc_samples_is_small_positive_int(self, K):
        assert isinstance(K.N_LC_SAMPLES, int)
        assert K.N_LC_SAMPLES > 0
        assert (
            K.N_LC_SAMPLES <= 10
        ), "N_LC_SAMPLES is unexpectedly large for a speed parameter"

    def test_n_window_size_is_300(self, K):
        """Thesis: fixed training window size = 300 activities."""
        assert K.N_WINDOW_SIZE == 300

    def test_n_windows_is_10(self, K):
        """Thesis: number of sliding windows = 10."""
        assert K.N_WINDOWS == 10

    def test_eval_on_val_is_true(self, K):
        """Default mode evaluates on validation set."""
        assert K.EVAL_ON_VAL is True

    def test_n_estimators_curve_less_than_500(self, K):
        """K uses a reduced tree count (100) for speed; G uses 500."""
        from G_outcome_tag_train import RF_PARAMS_BASE

        assert K.N_ESTIMATORS_CURVE < RF_PARAMS_BASE["n_estimators"]

    def test_n_estimators_curve_is_100(self, K):
        """Comment in file: 'RF trees for C/D (speed); G_outcome_tag_train uses 500'."""
        assert K.N_ESTIMATORS_CURVE == 100


# ===========================================================================
# M — data analysis eval set sizes
# ===========================================================================


class TestMSources:
    """Tests for the SOURCES dict in M_data_analysis_eval_set_sizes.py."""

    @pytest.fixture(scope="class")
    def sources(self):
        from M_data_analysis_eval_set_sizes import SOURCES

        return SOURCES

    def test_sources_has_exactly_4_keys(self, sources):
        assert len(sources) == 4

    @pytest.mark.parametrize(
        "key",
        [
            "overall_ratings",
            "outcome_tags",
            "cost_effectiveness",
            "ai_forecasting",
        ],
    )
    def test_sources_contains_key(self, sources, key):
        assert key in sources

    def test_sources_values_are_paths(self, sources):
        for key, val in sources.items():
            assert isinstance(
                val, Path
            ), f"SOURCES['{key}'] should be a Path, got {type(val)}"


class TestMLoadSplit:
    """Tests for the load_split() function in M_data_analysis_eval_set_sizes.py."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from M_data_analysis_eval_set_sizes import load_split

        self.load_split = load_split

    def test_returns_none_for_nonexistent_path(self, tmp_path):
        result = self.load_split(tmp_path / "nonexistent.csv", "test_label")
        assert result is None

    def test_returns_dataframe_for_valid_csv(self, tmp_path):
        csv = tmp_path / "valid.csv"
        csv.write_text("activity_id,split\nact_001,train\nact_002,val\n")
        result = self.load_split(csv, "test_label")
        assert isinstance(result, pd.DataFrame)

    def test_strips_whitespace_from_activity_id(self, tmp_path):
        csv = tmp_path / "whitespace.csv"
        csv.write_text("activity_id,split\n act_001 ,train\n  act_002  ,val\n")
        result = self.load_split(csv, "test_label")
        assert list(result["activity_id"]) == ["act_001", "act_002"]

    def test_activity_id_column_is_string_dtype(self, tmp_path):
        csv = tmp_path / "typed.csv"
        csv.write_text("activity_id,split\n12345,train\n67890,val\n")
        result = self.load_split(csv, "test_label")
        assert (
            result["activity_id"].dtype == object
        ), "activity_id should be read as string (object) dtype"

    def test_returned_dataframe_has_activity_id_column(self, tmp_path):
        csv = tmp_path / "cols.csv"
        csv.write_text("activity_id,split\nact_001,train\n")
        result = self.load_split(csv, "test_label")
        assert "activity_id" in result.columns
