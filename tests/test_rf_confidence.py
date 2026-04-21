"""
Tests for src/utils/overall_rating_rf_conformal.py

Focuses on the pure/lightweight functions that don't require
a trained model or large data:
  - get_error_bars_split_conformal
  - split_train_first_frac
  - module-level constants
"""

import math

import numpy as np
import pandas as pd
import pytest

import overall_rating_rf_conformal as rfc

# -- Module constants ----------------------------------------------------------


class TestConstants:
    def test_use_conformal_prediction_is_bool(self):
        assert isinstance(rfc.USE_CONFORMAL_PREDICTION, bool)

    def test_use_fixed_width_conformal_is_bool(self):
        assert isinstance(rfc.USE_FIXED_WIDTH_CONFORMAL, bool)

    def test_show_plots_is_false(self):
        # Tests should never pop up windows
        assert rfc.SHOW_PLOTS is False


# -- get_error_bars_split_conformal --------------------------------------------


class TestGetErrorBarsSplitConformal:
    @pytest.fixture
    def simple_series(self):
        rng = np.random.default_rng(0)
        idx = pd.RangeIndex(20)
        # Use random floats so .to_numpy() inside the function returns a fresh copy
        y_true = pd.Series(rng.uniform(0, 5, 20), index=idx)
        y_pred = pd.Series(rng.uniform(0, 5, 20), index=idx)
        cal_idx = idx[:10]
        return y_true, y_pred, cal_idx

    def test_returns_series(self, simple_series):
        y_true, y_pred, cal_idx = simple_series
        result = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred, cal_idx=cal_idx
        )
        assert isinstance(result, pd.Series)

    def test_same_index_as_y_pred(self, simple_series):
        y_true, y_pred, cal_idx = simple_series
        result = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred, cal_idx=cal_idx
        )
        assert result.index.equals(y_pred.index)

    def test_constant_value_across_index(self, simple_series):
        """Split conformal returns a constant half-width."""
        y_true, y_pred, cal_idx = simple_series
        result = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred, cal_idx=cal_idx
        )
        assert result.nunique() == 1

    def test_halfwidth_is_non_negative(self, simple_series):
        y_true, y_pred, cal_idx = simple_series
        result = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred, cal_idx=cal_idx
        )
        assert (result >= 0).all()

    def test_perfect_predictions_small_halfwidth(self):
        """Zero residuals on calibration set → q = 0 halfwidth."""
        idx = pd.RangeIndex(20)
        y_true = pd.Series(np.ones(20), index=idx)
        y_pred = pd.Series(np.ones(20), index=idx)  # perfect
        cal_idx = idx[:10]
        result = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred, cal_idx=cal_idx
        )
        assert float(result.iloc[0]) == pytest.approx(0.0)

    def test_larger_errors_give_larger_halfwidth(self):
        idx = pd.RangeIndex(20)
        y_true = pd.Series(np.zeros(20), index=idx)
        cal_idx = idx[:10]

        y_pred_small = pd.Series(np.full(20, 0.1), index=idx)
        y_pred_large = pd.Series(np.full(20, 2.0), index=idx)

        small_hw = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred_small, cal_idx=cal_idx
        ).iloc[0]
        large_hw = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred_large, cal_idx=cal_idx
        ).iloc[0]
        assert large_hw > small_hw

    def test_default_alpha_gives_90pct_name(self, simple_series):
        y_true, y_pred, cal_idx = simple_series
        result = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred, cal_idx=cal_idx, alpha=0.10
        )
        assert result.name == "pi90_halfwidth"

    def test_coverage_guarantee(self):
        """
        With alpha=0.10, at least 90% of calibration residuals should be
        <= the returned half-width (conformal guarantee).
        """
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.RangeIndex(n)
        y_true = pd.Series(rng.uniform(0, 5, n), index=idx)
        y_pred = pd.Series(rng.uniform(0, 5, n), index=idx)
        cal_idx = idx[:100]

        alpha = 0.10
        hw = rfc.get_error_bars_split_conformal(
            y_true=y_true, y_pred=y_pred, cal_idx=cal_idx, alpha=alpha
        ).iloc[0]

        residuals = (y_true.loc[cal_idx] - y_pred.loc[cal_idx]).abs()
        coverage = float((residuals <= hw).mean())
        assert coverage >= (1 - alpha) - 0.02  # allow tiny numerical slack
