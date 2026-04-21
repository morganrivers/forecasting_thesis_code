"""
Tests for C_overall_rating_insample_r2.py

Verifies that the code matches the thesis description for:
- fit_ols: statsmodels OLS fitting
- fit_ridge: sklearn RidgeCV+StandardScaler pipeline
- semi_partial_r2_ols: drop in R² when each feature is removed
- ridge_coef_contribution: beta_standardized * corr(x, y) decomposition
- compute_org_modes: mode per org from training set
- apply_org_modes: subtract org mode from y (demeaning / fixed effect)
"""

import numpy as np
import pandas as pd
import pytest

from C_overall_rating_insample_r2 import (
    apply_org_modes,
    compute_org_modes,
    fit_ols,
    fit_ridge,
    ridge_coef_contribution,
    semi_partial_r2_ols,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_linear_data(n=50, seed=0):
    """Return (X, y) where y = 2*x0 + 3*x1 + noise (small)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 0.1 * rng.standard_normal(n)
    return X, y


def _perfect_predictor_data(n=40, seed=1):
    """
    X has two columns: x0 perfectly predicts y, x1 is pure noise.
    y = x0 exactly.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    X = np.column_stack([x0, x1])
    y = x0.copy()
    return X, y


# ---------------------------------------------------------------------------
# TestFitOLS
# ---------------------------------------------------------------------------


class TestFitOLS:
    """Verify fit_ols uses statsmodels OLS and exposes the expected attributes."""

    def test_returns_fitted_result_with_rsquared(self):
        X, y = _simple_linear_data()
        result = fit_ols(X, y)
        assert hasattr(result, "rsquared"), "fit_ols result must have .rsquared"
        assert hasattr(result, "rsquared_adj"), "fit_ols result must have .rsquared_adj"

    def test_high_r2_for_nearly_linear_data(self):
        X, y = _simple_linear_data()
        result = fit_ols(X, y)
        # With very small noise and a true linear relationship R² should be near 1
        assert result.rsquared > 0.99

    def test_r2_between_0_and_1_for_valid_data(self):
        X, y = _simple_linear_data()
        result = fit_ols(X, y)
        assert 0.0 <= result.rsquared <= 1.0

    def test_adds_constant_internally(self):
        """fit_ols should add an intercept; check that n_params = p + 1."""
        n, p = 30, 3
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        result = fit_ols(X, y)
        # statsmodels includes intercept in params
        assert len(result.params) == p + 1

    def test_intercept_for_pure_noise_near_zero(self):
        """For centred noise data, intercept should be close to zero."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((100, 2))
        y = rng.standard_normal(100)
        result = fit_ols(X, y)
        # intercept is the first param (const)
        assert abs(result.params[0]) < 0.5  # loose bound; noise can push it slightly

    def test_adjusted_r2_leq_r2(self):
        """Adjusted R² should always be <= R² for p >= 1."""
        X, y = _simple_linear_data()
        result = fit_ols(X, y)
        assert result.rsquared_adj <= result.rsquared + 1e-10

    def test_perfect_fit_gives_r2_one(self):
        """A perfectly linear relationship (no noise) should give R² = 1."""
        n = 20
        X = np.linspace(0, 1, n).reshape(-1, 1)
        y = 5.0 * X[:, 0] + 2.0
        result = fit_ols(X, y)
        assert abs(result.rsquared - 1.0) < 1e-8

    def test_single_feature(self):
        """fit_ols should work with a single feature column."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 1))
        y = 2.0 * X[:, 0] + 0.1 * rng.standard_normal(50)
        result = fit_ols(X, y)
        assert result.rsquared > 0.9


# ---------------------------------------------------------------------------
# TestFitRidge
# ---------------------------------------------------------------------------


class TestFitRidge:
    """Verify fit_ridge uses a StandardScaler + RidgeCV pipeline."""

    def test_returns_pipeline_with_predict(self):
        X, y = _simple_linear_data()
        pipe = fit_ridge(X, y)
        assert hasattr(
            pipe, "predict"
        ), "fit_ridge must return an object with .predict()"

    def test_pipeline_has_standardscaler_and_ridgecv(self):
        X, y = _simple_linear_data()
        pipe = fit_ridge(X, y)
        assert "standardscaler" in pipe.named_steps
        assert "ridgecv" in pipe.named_steps

    def test_predictions_same_length_as_input(self):
        X, y = _simple_linear_data()
        pipe = fit_ridge(X, y)
        y_hat = pipe.predict(X)
        assert len(y_hat) == len(y)

    def test_alpha_chosen_from_logspace(self):
        """Default alphas span logspace(-3,4,60); chosen alpha must be in that range."""
        X, y = _simple_linear_data()
        pipe = fit_ridge(X, y)
        alpha = pipe.named_steps["ridgecv"].alpha_
        assert 1e-3 <= alpha <= 1e4

    def test_high_r2_for_nearly_linear_data(self):
        from sklearn.metrics import r2_score

        X, y = _simple_linear_data()
        pipe = fit_ridge(X, y)
        r2 = r2_score(y, pipe.predict(X))
        assert r2 > 0.99

    def test_custom_alphas_accepted(self):
        X, y = _simple_linear_data()
        custom_alphas = [0.1, 1.0, 10.0]
        pipe = fit_ridge(X, y, alphas=custom_alphas)
        alpha = pipe.named_steps["ridgecv"].alpha_
        assert alpha in custom_alphas


# ---------------------------------------------------------------------------
# TestSemiPartialR2OLS
# ---------------------------------------------------------------------------


class TestSemiPartialR2OLS:
    """
    Thesis description: semi-partial R² = R²_full - R²_without_j
    (refitting OLS without feature j and measuring the drop in R²).
    """

    def test_returns_dataframe_with_expected_columns(self):
        X, y = _simple_linear_data()
        result = fit_ols(X, y)
        r2_full = float(result.rsquared)
        df = semi_partial_r2_ols(X, y, ["feat_a", "feat_b"], r2_full)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "semi_partial_r2" in df.columns

    def test_one_row_per_feature(self):
        X, y = _simple_linear_data()
        feature_names = ["x0", "x1"]
        result = fit_ols(X, y)
        df = semi_partial_r2_ols(X, y, feature_names, float(result.rsquared))
        assert len(df) == len(feature_names)

    def test_semi_partial_r2_nonnegative_for_ols(self):
        """
        Thesis claim: removing a feature can only decrease or keep the same R².
        Therefore semi-partial R² = R²_full - R²_without_j >= 0 for OLS.
        This follows from the fact that OLS achieves the minimum residual sum of
        squares; adding more features cannot decrease in-sample R².
        """
        X, y = _simple_linear_data()
        result = fit_ols(X, y)
        r2_full = float(result.rsquared)
        df = semi_partial_r2_ols(X, y, ["x0", "x1"], r2_full)
        for _, row in df.iterrows():
            assert (
                row["semi_partial_r2"] >= -1e-9
            ), f"Semi-partial R² for {row['feature']} is negative: {row['semi_partial_r2']}"

    def test_perfect_predictor_has_high_semi_partial_r2(self):
        """
        x0 perfectly predicts y; removing x0 should yield a large drop in R².
        x1 is noise; removing x1 should cause a small drop.
        """
        X, y = _perfect_predictor_data()
        result = fit_ols(X, y)
        r2_full = float(result.rsquared)
        df = semi_partial_r2_ols(X, y, ["x0_perfect", "x1_noise"], r2_full)
        row_x0 = df[df["feature"] == "x0_perfect"].iloc[0]
        row_x1 = df[df["feature"] == "x1_noise"].iloc[0]
        # x0 is the perfect predictor: removing it loses almost all R²
        assert row_x0["semi_partial_r2"] > 0.5
        # x1 is pure noise: removing it has near-zero impact on R²
        assert abs(row_x1["semi_partial_r2"]) < 0.1

    def test_semi_partial_r2_matches_manual_calculation(self):
        """
        Manually compute R²_without_j and verify the function's output matches.
        """
        X, y = _simple_linear_data(n=60, seed=5)
        feature_names = ["a", "b"]
        result_full = fit_ols(X, y)
        r2_full = float(result_full.rsquared)

        df = semi_partial_r2_ols(X, y, feature_names, r2_full)

        # Manual: remove feature 0
        result_no0 = fit_ols(X[:, [1]], y)
        expected_drop_0 = r2_full - float(result_no0.rsquared)

        row_a = df[df["feature"] == "a"].iloc[0]
        assert abs(row_a["semi_partial_r2"] - expected_drop_0) < 1e-10

        # Manual: remove feature 1
        result_no1 = fit_ols(X[:, [0]], y)
        expected_drop_1 = r2_full - float(result_no1.rsquared)

        row_b = df[df["feature"] == "b"].iloc[0]
        assert abs(row_b["semi_partial_r2"] - expected_drop_1) < 1e-10

    def test_sorted_by_semi_partial_r2_descending(self):
        X, y = _simple_linear_data()
        result = fit_ols(X, y)
        df = semi_partial_r2_ols(X, y, ["x0", "x1"], float(result.rsquared))
        vals = df["semi_partial_r2"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_single_feature_semi_partial_equals_full_r2(self):
        """With one feature, removing it must give R² = 0 (intercept-only model)."""
        rng = np.random.default_rng(22)
        X = rng.standard_normal((50, 1))
        y = 3.0 * X[:, 0] + 0.05 * rng.standard_normal(50)
        result = fit_ols(X, y)
        r2_full = float(result.rsquared)
        df = semi_partial_r2_ols(X, y, ["only_feat"], r2_full)
        # Removing the only feature => intercept-only OLS => R² = 0
        # so semi_partial_r2 ≈ r2_full
        assert abs(df.iloc[0]["semi_partial_r2"] - r2_full) < 1e-8


# ---------------------------------------------------------------------------
# TestRidgeCoefContribution
# ---------------------------------------------------------------------------


class TestRidgeCoefContribution:
    """
    Thesis formula: contribution_j = beta_standardized_j * corr(x_j_std, y)
    where beta_standardized comes from the Ridge model applied to scaled X.
    """

    def test_returns_dataframe_with_expected_columns(self):
        X, y = _simple_linear_data()
        pipe = fit_ridge(X, y)
        df = ridge_coef_contribution(X, y, ["f0", "f1"], pipe)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "coef_contribution" in df.columns

    def test_one_row_per_feature(self):
        X, y = _simple_linear_data()
        pipe = fit_ridge(X, y)
        feature_names = ["f0", "f1"]
        df = ridge_coef_contribution(X, y, feature_names, pipe)
        assert len(df) == len(feature_names)

    def test_sorted_descending(self):
        X, y = _simple_linear_data()
        pipe = fit_ridge(X, y)
        df = ridge_coef_contribution(X, y, ["f0", "f1"], pipe)
        vals = df["coef_contribution"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_formula_matches_manual_computation(self):
        """
        Verify: contribution_j = ridge.coef_[j] * corr(X_std[:, j], y)
        using the scaler stored in the pipeline.
        """
        X, y = _simple_linear_data(n=80, seed=10)
        pipe = fit_ridge(X, y)
        df = ridge_coef_contribution(X, y, ["f0", "f1"], pipe)

        scaler = pipe.named_steps["standardscaler"]
        ridge = pipe.named_steps["ridgecv"]
        X_std = scaler.transform(X)

        for i, name in enumerate(["f0", "f1"]):
            beta = ridge.coef_.ravel()[i]
            corr = float(np.corrcoef(X_std[:, i], y)[0, 1])
            expected = beta * corr
            row = df[df["feature"] == name].iloc[0]
            assert (
                abs(row["coef_contribution"] - expected) < 1e-12
            ), f"Feature {name}: expected {expected}, got {row['coef_contribution']}"

    def test_sum_positive_for_predictive_features(self):
        """For a well-fitted ridge model on near-linear data, sum of contributions > 0."""
        X, y = _simple_linear_data(n=100, seed=3)
        pipe = fit_ridge(X, y)
        df = ridge_coef_contribution(X, y, ["f0", "f1"], pipe)
        assert df["coef_contribution"].sum() > 0

    def test_perfectly_predictive_feature_has_high_contribution(self):
        """
        x0 is the sole signal; x1 is noise.
        Contribution of x0 must exceed that of x1.
        """
        X, y = _perfect_predictor_data(n=60)
        pipe = fit_ridge(X, y)
        df = ridge_coef_contribution(X, y, ["x0_signal", "x1_noise"], pipe)
        contrib_signal = df[df["feature"] == "x0_signal"].iloc[0]["coef_contribution"]
        contrib_noise = df[df["feature"] == "x1_noise"].iloc[0]["coef_contribution"]
        assert contrib_signal > contrib_noise


# ---------------------------------------------------------------------------
# TestComputeOrgModes
# ---------------------------------------------------------------------------


class TestComputeOrgModes:
    """
    compute_org_modes(df_fit, org_cols, y_col='rating') should:
    - Return (overall_mode, {org_col: mode_for_that_org})
    - overall_mode is the mode of all ratings in df_fit
    - per-org mode is computed from the subset where org_col == 1
    """

    def _make_df(self):
        """DataFrame with two org columns and ratings."""
        return pd.DataFrame(
            {
                "rep_org_A": [1, 1, 1, 0, 0, 0, 0],
                "rep_org_B": [0, 0, 0, 1, 1, 1, 0],
                "rating": [3, 3, 4, 5, 5, 5, 3],
            }
        )

    def test_returns_tuple_of_float_and_dict(self):
        df = self._make_df()
        org_cols = ["rep_org_A", "rep_org_B"]
        overall_mode, org_modes = compute_org_modes(df, org_cols)
        assert isinstance(overall_mode, float)
        assert isinstance(org_modes, dict)

    def test_overall_mode_correct(self):
        df = self._make_df()
        # ratings: [3, 3, 4, 5, 5, 5, 3] -> mode is 3 (appears 3 times) or 5 (3 times)
        # pandas mode returns the smallest when tied
        org_cols = ["rep_org_A", "rep_org_B"]
        overall_mode, _ = compute_org_modes(df, org_cols)
        assert overall_mode in [3.0, 5.0]  # either is a valid mode

    def test_per_org_mode_correct(self):
        df = self._make_df()
        org_cols = ["rep_org_A", "rep_org_B"]
        _, org_modes = compute_org_modes(df, org_cols)
        # rep_org_A rows: ratings 3, 3, 4 -> mode = 3
        assert org_modes["rep_org_A"] == 3.0
        # rep_org_B rows: ratings 5, 5, 5 -> mode = 5
        assert org_modes["rep_org_B"] == 5.0

    def test_org_cols_present_in_dict(self):
        df = self._make_df()
        org_cols = ["rep_org_A", "rep_org_B"]
        _, org_modes = compute_org_modes(df, org_cols)
        assert "rep_org_A" in org_modes
        assert "rep_org_B" in org_modes

    def test_missing_org_falls_back_to_overall_mode(self):
        """If no rows belong to an org column, fallback to overall_mode."""
        df = pd.DataFrame(
            {
                "rep_org_A": [1, 1, 0, 0],
                "rep_org_B": [0, 0, 0, 0],  # empty org
                "rating": [4, 4, 3, 3],
            }
        )
        org_cols = ["rep_org_A", "rep_org_B"]
        overall_mode, org_modes = compute_org_modes(df, org_cols)
        # rep_org_B has no members -> fallback to overall_mode
        assert org_modes["rep_org_B"] == overall_mode

    def test_mode_uses_y_col_parameter(self):
        """compute_org_modes should respect y_col argument."""
        df = pd.DataFrame(
            {
                "rep_org_X": [1, 1, 1, 0],
                "custom_target": [2, 2, 2, 1],
                "rating": [5, 5, 5, 1],
            }
        )
        _, org_modes_custom = compute_org_modes(
            df, ["rep_org_X"], y_col="custom_target"
        )
        assert org_modes_custom["rep_org_X"] == 2.0

    def test_returns_float_types(self):
        df = self._make_df()
        overall_mode, org_modes = compute_org_modes(df, ["rep_org_A", "rep_org_B"])
        assert isinstance(overall_mode, float)
        for v in org_modes.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# TestApplyOrgModes
# ---------------------------------------------------------------------------


class TestApplyOrgModes:
    """
    apply_org_modes(df_split, overall_mode, org_modes, org_cols) should:
    - Return a pd.Series of org-mode per row
    - Activities belonging to org X get org_modes[X]
    - Activities in no org get overall_mode
    """

    def _make_df_split(self):
        return pd.DataFrame(
            {
                "rep_org_A": [1, 0, 0],
                "rep_org_B": [0, 1, 0],
            }
        )

    def test_returns_series(self):
        df = self._make_df_split()
        pom = apply_org_modes(
            df, 3.0, {"rep_org_A": 4.0, "rep_org_B": 5.0}, ["rep_org_A", "rep_org_B"]
        )
        assert isinstance(pom, pd.Series)

    def test_same_length_as_input(self):
        df = self._make_df_split()
        pom = apply_org_modes(
            df, 3.0, {"rep_org_A": 4.0, "rep_org_B": 5.0}, ["rep_org_A", "rep_org_B"]
        )
        assert len(pom) == len(df)

    def test_org_a_gets_org_a_mode(self):
        df = self._make_df_split()
        pom = apply_org_modes(
            df, 3.0, {"rep_org_A": 4.0, "rep_org_B": 5.0}, ["rep_org_A", "rep_org_B"]
        )
        assert pom.iloc[0] == 4.0

    def test_org_b_gets_org_b_mode(self):
        df = self._make_df_split()
        pom = apply_org_modes(
            df, 3.0, {"rep_org_A": 4.0, "rep_org_B": 5.0}, ["rep_org_A", "rep_org_B"]
        )
        assert pom.iloc[1] == 5.0

    def test_no_org_gets_overall_mode(self):
        df = self._make_df_split()
        pom = apply_org_modes(
            df, 3.0, {"rep_org_A": 4.0, "rep_org_B": 5.0}, ["rep_org_A", "rep_org_B"]
        )
        assert pom.iloc[2] == 3.0

    def test_index_preserved(self):
        """The returned Series should share the DataFrame's index."""
        df = pd.DataFrame(
            {"rep_org_A": [1, 0], "rep_org_B": [0, 1]},
            index=[10, 20],
        )
        pom = apply_org_modes(
            df, 3.0, {"rep_org_A": 4.0, "rep_org_B": 5.0}, ["rep_org_A", "rep_org_B"]
        )
        assert list(pom.index) == [10, 20]

    def test_demeaning_and_inverse(self):
        """
        Thesis describes: y_demeaned = y - mode_org
        Inverse:          y_corrected = prediction + mode_org

        Verify that subtracting and adding back the org mode is an identity.
        """
        df = pd.DataFrame(
            {
                "rep_org_A": [1, 1, 0, 0],
                "rep_org_B": [0, 0, 1, 1],
                "rating": [4.0, 5.0, 3.0, 3.0],
            }
        )
        org_cols = ["rep_org_A", "rep_org_B"]
        overall_mode, org_modes = compute_org_modes(df, org_cols)
        pom = apply_org_modes(df, overall_mode, org_modes, org_cols)

        y_raw = df["rating"].astype(float)
        y_demeaned = y_raw - pom
        y_reconstructed = y_demeaned + pom

        np.testing.assert_allclose(y_reconstructed.values, y_raw.values)

    def test_fillna_handles_missing_org_col(self):
        """
        org_col values may contain NaN (the function uses fillna(0)).
        NaN rows should fall back to overall_mode (treated as not belonging to that org).
        """
        df = pd.DataFrame(
            {
                "rep_org_A": [1.0, np.nan, 0.0],
                "rep_org_B": [0.0, 1.0, np.nan],
            }
        )
        pom = apply_org_modes(
            df, 9.0, {"rep_org_A": 4.0, "rep_org_B": 5.0}, ["rep_org_A", "rep_org_B"]
        )
        assert pom.iloc[0] == 4.0
        assert pom.iloc[1] == 5.0
        # row 2 has both 0 and NaN -> NaN filled to 0 -> no org -> overall_mode
        assert pom.iloc[2] == 9.0


# ---------------------------------------------------------------------------
# TestComputeAndApplyOrgModesRoundtrip
# ---------------------------------------------------------------------------


class TestComputeAndApplyOrgModesRoundtrip:
    """Integration tests for the compute/apply pair together."""

    def test_compute_apply_roundtrip(self):
        """compute_org_modes on a training set, then apply to same set is identity add/subtract."""
        df_train = pd.DataFrame(
            {
                "rep_org_WB": [1, 1, 1, 0, 0],
                "rep_org_DE": [0, 0, 0, 1, 1],
                "rating": [3.0, 3.0, 4.0, 5.0, 5.0],
            }
        )
        org_cols = ["rep_org_WB", "rep_org_DE"]
        overall_mode, org_modes = compute_org_modes(df_train, org_cols)
        pom_train = apply_org_modes(df_train, overall_mode, org_modes, org_cols)

        y_train = df_train["rating"].astype(float)
        y_delta = y_train - pom_train
        y_recovered = y_delta + pom_train
        np.testing.assert_allclose(y_recovered.values, y_train.values)

    def test_modes_consistent_with_per_org_most_common_value(self):
        """The mode returned for each org must equal the most frequent rating for that org."""
        ratings_wb = [3, 3, 3, 4, 5]
        ratings_de = [5, 5, 4]
        all_ratings = ratings_wb + ratings_de + [3]  # 1 activity with no org
        df = pd.DataFrame(
            {
                "rep_org_WB": [1] * 5 + [0] * 3 + [0],
                "rep_org_DE": [0] * 5 + [1] * 3 + [0],
                "rating": all_ratings,
            }
        )
        org_cols = ["rep_org_WB", "rep_org_DE"]
        _, org_modes = compute_org_modes(df, org_cols)
        assert org_modes["rep_org_WB"] == 3.0  # 3 appears 3 times for WB
        assert org_modes["rep_org_DE"] == 5.0  # 5 appears 2 times for DE


# ---------------------------------------------------------------------------
# TestSemiPartialR2Properties
# ---------------------------------------------------------------------------


class TestSemiPartialR2Properties:
    """Additional property-based tests for semi-partial R²."""

    def test_sum_of_semi_partial_r2_leq_full_r2(self):
        """
        Due to collinearity, the sum of semi-partial R² values may exceed the
        full model R² in theory (multicollinarity can make individual drops
        look small), but for independent features the sum should approximate full R².
        This test uses near-orthogonal features to verify the sum is plausible.
        """
        # Build near-orthogonal features
        rng = np.random.default_rng(42)
        n = 200
        # Use QR decomposition to guarantee orthogonal columns
        Q, _ = np.linalg.qr(rng.standard_normal((n, 3)))
        X = Q  # exactly orthogonal columns
        y = 2.0 * X[:, 0] + 1.5 * X[:, 1] + 0.5 * X[:, 2] + 0.1 * rng.standard_normal(n)
        result = fit_ols(X, y)
        r2_full = float(result.rsquared)
        df = semi_partial_r2_ols(X, y, ["q0", "q1", "q2"], r2_full)
        total_spr2 = df["semi_partial_r2"].sum()
        # For orthogonal features, sum of semi-partial R² ≈ full R²
        # (each feature's contribution doesn't overlap)
        assert abs(total_spr2 - r2_full) < 0.05  # within 5 percentage points

    def test_irrelevant_feature_has_near_zero_semi_partial_r2(self):
        """Adding a completely uncorrelated random feature should have near-zero semi-partial R²."""
        rng = np.random.default_rng(9)
        n = 100
        x_signal = rng.standard_normal(n)
        x_noise = rng.standard_normal(n)
        # Make them uncorrelated by construction via Gram-Schmidt
        x_noise = (
            x_noise - np.dot(x_noise, x_signal) / np.dot(x_signal, x_signal) * x_signal
        )
        X = np.column_stack([x_signal, x_noise])
        y = 3.0 * x_signal + 0.1 * rng.standard_normal(n)

        result = fit_ols(X, y)
        r2_full = float(result.rsquared)
        df = semi_partial_r2_ols(X, y, ["signal", "noise"], r2_full)
        noise_spr2 = df[df["feature"] == "noise"].iloc[0]["semi_partial_r2"]
        assert abs(noise_spr2) < 0.05
