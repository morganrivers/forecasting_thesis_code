"""
Tests for overall_rating_models.py and llm_grading_utils.calculate_metrics.

overall_rating_models wraps scikit-learn/XGBoost estimators with median
imputation and an optional Extra Trees ensemble (thesis §RF_ET_ensemble).
The LLM Adjustment Ridge Regression (thesis §LLM_adjustment) uses alpha=5
and clips predictions to [0, 5].  bootstrap_ci is used for reporting
confidence intervals on held-out metrics.

calculate_metrics (llm_grading_utils) computes RMSE, R², MAE, and binary
side-accuracy (threshold 3.5 on 0-5 scale) for graded forecast outputs.
"""

import numpy as np
import pandas as pd
import pytest

import ml_models as orm
from llm_grading_utils import calculate_metrics

FEAT_COLS = ["feat_a", "feat_b", "feat_c"]


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def _mse(self, a, b):
        return float(np.mean((a - b) ** 2))

    def test_returns_three_keys(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0, 5, 50)
        yhat = y + rng.normal(0, 0.3, 50)
        result = orm.bootstrap_ci(y, yhat, self._mse, n_bootstrap=100)
        assert set(result.keys()) >= {"mean", "lower", "upper"}

    def test_lower_le_mean_le_upper(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0, 5, 60)
        yhat = y + rng.normal(0, 0.5, 60)
        result = orm.bootstrap_ci(y, yhat, self._mse, n_bootstrap=200)
        assert result["lower"] <= result["mean"]
        assert result["mean"] <= result["upper"]

    def test_lower_le_upper(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(0, 5, 80)
        yhat = rng.uniform(0, 5, 80)
        result = orm.bootstrap_ci(y, yhat, self._mse, n_bootstrap=200)
        assert result["lower"] <= result["upper"]

    def test_too_few_samples_returns_nan(self):
        result = orm.bootstrap_ci(np.array([1.0]), np.array([1.0]), self._mse)
        assert np.isnan(result["mean"])
        assert np.isnan(result["lower"])
        assert np.isnan(result["upper"])

    def test_nan_pairs_removed_before_bootstrap(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 10)
        yhat = np.array([1.1, 2.1, np.nan, 4.1, 5.1] * 10)
        result = orm.bootstrap_ci(y, yhat, self._mse, n_bootstrap=100)
        assert np.isfinite(result["mean"])

    def test_perfect_metric_value_near_zero_mse(self):
        y = np.arange(1, 51, dtype=float)
        result = orm.bootstrap_ci(y, y.copy(), self._mse, n_bootstrap=100)
        assert result["mean"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# run_ridge_glm_median_impute_noclip
# ---------------------------------------------------------------------------


class TestRunRidgeGLM:
    """Ridge regressor with median imputation and train-only fitting."""

    def test_output_length_matches_data(self, feature_df):
        y_hat, _ = orm.run_ridge_glm_median_impute_noclip(feature_df, FEAT_COLS)
        assert len(y_hat) == len(feature_df)

    def test_model_has_medians_attribute(self, feature_df):
        _, model = orm.run_ridge_glm_median_impute_noclip(feature_df, FEAT_COLS)
        assert hasattr(model, "medians_")

    def test_predictions_are_finite(self, feature_df):
        y_hat, _ = orm.run_ridge_glm_median_impute_noclip(feature_df, FEAT_COLS)
        assert np.all(np.isfinite(y_hat))

    def test_train_index_restricts_fitting(self, feature_df):
        n = len(feature_df)
        train_idx = feature_df.index[:30]
        y_hat_full, _ = orm.run_ridge_glm_median_impute_noclip(feature_df, FEAT_COLS)
        y_hat_sub, _ = orm.run_ridge_glm_median_impute_noclip(
            feature_df, FEAT_COLS, train_index=train_idx
        )
        # Same data, different training set → predictions differ
        assert not np.allclose(y_hat_full, y_hat_sub)

    def test_nan_features_handled(self, feature_df):
        # feature_df already has NaNs in feat_a and feat_b — must not raise
        y_hat, _ = orm.run_ridge_glm_median_impute_noclip(feature_df, FEAT_COLS)
        assert len(y_hat) == len(feature_df)

    def test_simple_linear_relationship_learned(self):
        # y = 2 * x1 + noise; ridge should predict reasonably well
        rng = np.random.default_rng(42)
        n = 80
        x = rng.uniform(0, 5, n)
        df = pd.DataFrame({"feat_a": x, "rating": 2 * x + rng.normal(0, 0.1, n)})
        y_hat, _ = orm.run_ridge_glm_median_impute_noclip(df, ["feat_a"])
        corr = float(np.corrcoef(y_hat, df["rating"])[0, 1])
        assert corr > 0.95


# ---------------------------------------------------------------------------
# run_random_forest_median_impute_noclip
# ---------------------------------------------------------------------------


class TestRunRandomForest:
    """RF (+ optional Extra Trees) regressor — thesis §RF_ET_ensemble."""

    def test_output_length_matches_data(self, feature_df):
        y_hat, _ = orm.run_random_forest_median_impute_noclip(
            feature_df, FEAT_COLS, rf_params={"n_estimators": 20}
        )
        assert len(y_hat) == len(feature_df)

    def test_predictions_are_finite(self, feature_df):
        y_hat, _ = orm.run_random_forest_median_impute_noclip(
            feature_df, FEAT_COLS, rf_params={"n_estimators": 20}
        )
        assert np.all(np.isfinite(y_hat))

    def test_ensemble_with_extra_trees(self, feature_df):
        y_hat, rf = orm.run_random_forest_median_impute_noclip(
            feature_df,
            FEAT_COLS,
            rf_params={"n_estimators": 20},
            ensemble_with_extratrees=True,
        )
        assert len(y_hat) == len(feature_df)
        assert np.all(np.isfinite(y_hat))

    def test_return_extra_flag(self, feature_df):
        result = orm.run_random_forest_median_impute_noclip(
            feature_df,
            FEAT_COLS,
            rf_params={"n_estimators": 10},
            ensemble_with_extratrees=True,
            return_extra=True,
        )
        assert len(result) == 3  # (y_hat, rf, extra_trees)

    def test_nan_features_handled(self, feature_df):
        y_hat, _ = orm.run_random_forest_median_impute_noclip(
            feature_df, FEAT_COLS, rf_params={"n_estimators": 10}
        )
        assert len(y_hat) == len(feature_df)

    def test_train_index_used(self, feature_df):
        train_idx = feature_df.index[:25]
        y_hat_full, _ = orm.run_random_forest_median_impute_noclip(
            feature_df, FEAT_COLS, rf_params={"n_estimators": 10}
        )
        y_hat_sub, _ = orm.run_random_forest_median_impute_noclip(
            feature_df,
            FEAT_COLS,
            train_index=train_idx,
            rf_params={"n_estimators": 10},
        )
        assert not np.allclose(y_hat_full, y_hat_sub)


# ---------------------------------------------------------------------------
# apply_start_year_trend_correction
# ---------------------------------------------------------------------------


class TestApplyStartYearTrendCorrection:
    """Thesis §start_year_trend_correction: ridge α=50 absorbs linear temporal drift."""

    def _make_data(self, n=50):
        rng = np.random.default_rng(9)
        dates = pd.date_range("2005-01-01", periods=n, freq="180D")
        df = pd.DataFrame(
            {
                "start_date": dates,
                "pred_rf_llm_modded": rng.uniform(1, 5, n),
            },
            index=pd.RangeIndex(n),
        )
        y = pd.Series(rng.uniform(0, 5, n), index=df.index)
        return df, y

    def test_modifies_pred_col_in_place(self):
        df, y = self._make_data()
        original = df["pred_rf_llm_modded"].copy()
        train_idx = df.index[:30]
        orm.apply_start_year_trend_correction(df, y, train_idx)
        assert not df["pred_rf_llm_modded"].equals(original)

    def test_all_rows_corrected(self):
        df, y = self._make_data()
        train_idx = df.index[:30]
        orm.apply_start_year_trend_correction(df, y, train_idx)
        assert df["pred_rf_llm_modded"].notna().all()

    def test_zero_residuals_produce_near_zero_correction(self):
        # If y == pred, residuals are zero → correction ≈ 0
        n = 40
        rng = np.random.default_rng(11)
        dates = pd.date_range("2005-01-01", periods=n, freq="180D")
        df = pd.DataFrame(
            {
                "start_date": dates,
                "pred_rf_llm_modded": rng.uniform(1, 4, n),
            },
            index=pd.RangeIndex(n),
        )
        y = df["pred_rf_llm_modded"].copy()
        original = df["pred_rf_llm_modded"].copy()
        train_idx = df.index[:25]
        orm.apply_start_year_trend_correction(df, y, train_idx, alpha=50.0)
        # With α=50 and zero residuals, correction should be very small
        max_diff = (df["pred_rf_llm_modded"] - original).abs().max()
        assert max_diff < 0.5

    def test_custom_pred_col_name(self):
        df, y = self._make_data()
        df["my_preds"] = df["pred_rf_llm_modded"].copy()
        original = df["my_preds"].copy()
        train_idx = df.index[:30]
        orm.apply_start_year_trend_correction(df, y, train_idx, pred_col="my_preds")
        assert not df["my_preds"].equals(original)


# ---------------------------------------------------------------------------
# calculate_metrics  (llm_grading_utils)
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    """Thesis §scoring_metrics: RMSE, R², MAE, binary accuracy at threshold 3.5."""

    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_metrics(y, y.copy())
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["r2"] == pytest.approx(1.0, abs=1e-10)
        assert result["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_returns_all_keys(self):
        y = np.array([1.0, 3.0, 5.0])
        result = calculate_metrics(y, y.copy())
        assert set(result.keys()) >= {"rmse", "r2", "mae", "side_accuracy_3_5", "n"}

    def test_n_equals_input_length(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = calculate_metrics(y, y.copy())
        assert result["n"] == 4

    def test_n_excludes_nan_pairs(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        yhat = np.array([1.0, 2.0, 3.0, 4.0])
        result = calculate_metrics(y, yhat)
        assert result["n"] == 3

    def test_empty_arrays_return_nan(self):
        result = calculate_metrics(np.array([]), np.array([]))
        assert np.isnan(result["rmse"])
        assert np.isnan(result["r2"])
        assert np.isnan(result["mae"])
        assert result["n"] == 0

    def test_all_nan_returns_nan(self):
        y = np.array([np.nan, np.nan])
        result = calculate_metrics(y, y.copy())
        assert result["n"] == 0

    def test_constant_predictions_r2_near_zero(self):
        # Predicting the mean gives R² = 0
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yhat = np.full(5, y.mean())
        result = calculate_metrics(y, yhat)
        assert result["r2"] == pytest.approx(0.0, abs=1e-10)

    def test_side_accuracy_threshold_3_5(self):
        # All above 3.5 correctly predicted as above
        y_true = np.array([4.0, 4.5, 5.0])
        y_pred = np.array([3.6, 4.1, 4.8])
        result = calculate_metrics(y_true, y_pred)
        assert result["side_accuracy_3_5"] == pytest.approx(1.0)

    def test_side_accuracy_wrong_side(self):
        # True above 3.5 but predicted below → 0% accuracy
        y_true = np.array([4.0, 4.5, 5.0])
        y_pred = np.array([2.0, 1.5, 3.0])
        result = calculate_metrics(y_true, y_pred)
        assert result["side_accuracy_3_5"] == pytest.approx(0.0)

    def test_rmse_known_value(self):
        # Constant +0.5 bias → RMSE = 0.5
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = calculate_metrics(y, y + 0.5)
        assert result["rmse"] == pytest.approx(0.5)

    def test_lists_accepted(self):
        result = calculate_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert result["rmse"] == pytest.approx(0.0)
