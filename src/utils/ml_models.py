"""
Model wrappers and training utilities for OLS, Ridge, Random Forest, XGBoost,
and ordinal regression used in overall rating prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


DISABLE_ET = False

BUCKET_EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])  # 0..5 buckets


CACHE = Path("../../data/bayesian_linear_cache.nc")
BUCKET_EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])  # 6 buckets: 0..5


def run_ridge_glm_median_impute_noclip(
    data,
    feature_cols,
    target_col="rating",
    train_index=None,
    alphas=None,
):
    """
    Ridge replacement for run_statsmodels_glm_median_impute.

    - Fit only on rows in `train_index` (if given).
    - Median impute using TRAINING medians (same as your GLM helper).
    - Standardize using TRAINING stats.
    - Choose alpha via RidgeCV on TRAIN.
    - Return predictions for ALL rows in `data` in original order.
    """
    if train_index is None:
        train_index = data.index

    if alphas is None:
        alphas = np.logspace(-3, 3, 40)

    train = data.loc[train_index]
    X_train = train[feature_cols].copy()
    y_train = train[target_col].astype(float)

    # medians from TRAINING only (match GLM helper behavior)
    medians = X_train.median(numeric_only=True)

    X_train_imp = X_train.fillna(medians)
    X_all_imp = data[feature_cols].copy().fillna(medians)

    # ridge with scaling (fit on TRAIN only)
    model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=alphas, fit_intercept=True),
    )
    model.fit(X_train_imp.to_numpy(dtype=float), y_train.to_numpy(dtype=float))

    y_hat_all = model.predict(X_all_imp.to_numpy(dtype=float))

    # keep medians around if you ever need to reproduce preprocessing
    model.medians_ = medians

    return y_hat_all, model


def run_random_forest_median_impute_noclip(
    data,
    feature_cols,
    target_col="rating",
    train_index=None,
    rf_params=None,
    sample_weight=None,
    ensemble_with_extratrees=False,
    return_extra=False,
):
    """
    Same as run_random_forest_median_impute but without clipping predictions to [0, 5].
    If ensemble_with_extratrees=True, averages RF and ExtraTrees predictions.
    """
    from sklearn.ensemble import ExtraTreesRegressor

    if DISABLE_ET:
        ensemble_with_extratrees = False

    if train_index is None:
        train_index = data.index

    train = data.loc[train_index]

    X_train = train[feature_cols].copy()
    y_train = train[target_col].astype(float)

    medians = X_train.median(numeric_only=True)

    X_train_imp = X_train.fillna(medians)
    X_all_imp = data[feature_cols].copy().fillna(medians)

    params = dict(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )
    if rf_params:
        params.update(rf_params)

    rf = RandomForestRegressor(**params)
    rf.fit(X_train_imp, y_train, sample_weight=sample_weight)
    y_hat_all = rf.predict(X_all_imp)

    extra = None
    if ensemble_with_extratrees:
        extra_params = {k: v for k, v in params.items()}
        extra_params["random_state"] = 43
        extra = ExtraTreesRegressor(**extra_params)
        extra.fit(X_train_imp, y_train, sample_weight=sample_weight)
        y_hat_all = (y_hat_all + extra.predict(X_all_imp)) / 2.0

    if return_extra:
        return y_hat_all, rf, extra
    return y_hat_all, rf


def apply_start_year_trend_correction(
    data,
    y,
    train_idx,
    pred_col="pred_rf_llm_modded",
    alpha=50.0,
):
    """
    Fit Ridge(alpha) on residual ~ start_year using train_idx rows, then add
    the fitted trend to pred_col for all rows.  Modifies data[pred_col] in-place.

    This absorbs any linear temporal drift in the model's calibration without
    touching pairwise ordering (POP is unchanged by a monotone level shift).
    """
    train_mask = data.index.intersection(pd.Index(train_idx))
    train_df = data.loc[train_mask].copy()
    train_df = train_df.dropna(subset=[pred_col, "start_date"])
    train_df = train_df.loc[y.reindex(train_df.index).notna()]

    years_tr = train_df["start_date"].dt.year.astype(float).to_numpy().reshape(-1, 1)
    resid_tr = (
        y.loc[train_df.index].astype(float) - train_df[pred_col].astype(float)
    ).to_numpy()

    yr_ridge = Ridge(alpha=alpha, fit_intercept=True)
    yr_ridge.fit(years_tr, resid_tr)
    slope = float(yr_ridge.coef_[0])
    intercept = float(yr_ridge.intercept_)

    print(
        f"\nStart-year trend correction fitted on n={len(train_df)} train rows (Ridge alpha={alpha})."
    )
    print(f"  residual ~ start_year:  slope={slope:+.4f}/yr  intercept={intercept:.4f}")

    years_all = data["start_date"].dt.year.astype(float)
    correction = intercept + slope * years_all
    data[pred_col] = data[pred_col].astype(float) + correction


def run_xgboost_native_missing(
    data,
    feature_cols,
    target_col="rating",
    train_index=None,
    xgb_params=None,
    early_stopping_rounds=50,
    early_stopping_val_frac=0.15,
    early_stopping_seed=42,
    clip_pred=True,
):
    """
    Fit an XGBoost regressor without median imputation.

    XGBoost natively handles NaN values by learning the optimal default
    direction (left or right branch) for each split at training time.
    Passing raw NaN values is strictly better than median imputation for
    tree-based models because it avoids polluting the split statistics.

    - Fit only on rows in `train_index` (if given).
    - Uses an internal holdout to find best_iteration via early stopping,
      then retrains on ALL of train_index with that fixed n_estimators.
    - Set early_stopping_rounds=None to skip early stopping entirely.
    - Returns (y_hat_all, xgb_model) aligned with data.index.
    """
    from sklearn.model_selection import train_test_split

    if train_index is None:
        train_index = data.index

    train = data.loc[train_index]

    X_train = train[feature_cols].copy().astype(float)
    y_train = train[target_col].astype(float)
    X_all = data[feature_cols].copy().astype(float)

    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    if xgb_params:
        params.update(xgb_params)

    if early_stopping_rounds is not None:
        # Phase 1: find best_iteration on holdout
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=early_stopping_val_frac,
            random_state=early_stopping_seed,
        )
        tuning_params = {**params, "early_stopping_rounds": early_stopping_rounds}
        tuning_model = xgb.XGBRegressor(**tuning_params)
        tuning_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        best_n = tuning_model.best_iteration + 1
        # print(f"XGBoost early stopping found best_iteration={best_n}; retraining on full train set")

        # Phase 2: retrain on full train with that fixed count
        params["n_estimators"] = best_n
        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_train, y_train)
    else:
        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_train, y_train)

    y_hat_all = xgb_model.predict(X_all)
    if clip_pred:
        y_hat_all = np.clip(y_hat_all, 0.0, 5.0)
    return y_hat_all, xgb_model


########## Using org-specific ratings and ordinal models ##############
CATS_0_5 = np.asarray([0, 1, 2, 3, 4, 5], dtype=float)


def one_sd_shift_importance(
    *,
    model,
    data: pd.DataFrame,
    feature_cols: list[str],
    train_idx,
):
    """
    Cheap, model-based local importance at a 'typical' point:
      importance_abs_1sd_j = | f(x0 + sd_j e_j) - f(x0 - sd_j e_j) | / 2
    Uses TRAIN median for baseline x0 and TRAIN-imputed SDs.
    """
    tr = pd.Index(train_idx)

    Xtr = data.loc[tr, feature_cols].astype(float)
    med = Xtr.median(numeric_only=True)
    Xtr_imp = Xtr.fillna(med)
    sd = Xtr_imp.std(numeric_only=True).replace(0.0, np.nan)

    x0 = med.reindex(feature_cols).to_frame().T
    x0 = x0.fillna(med)  # in case a feature is all-NaN in train

    pred0 = float(model.predict(x0)[0])

    rows = []
    for f in feature_cols:
        s = float(sd.get(f, np.nan))
        if not np.isfinite(s):
            continue

        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[f] = float(x_plus[f].iloc[0]) + s
        x_minus[f] = float(x_minus[f].iloc[0]) - s

        p_plus = float(model.predict(x_plus)[0])
        p_minus = float(model.predict(x_minus)[0])

        delta = (p_plus - p_minus) / 2.0
        rows.append(
            {
                "feature": f,
                "sd_train": s,
                "pred0": pred0,
                "delta_pred_1sd": delta,  # signed
                "importance_abs_1sd": abs(delta),  # magnitude
            }
        )

    out = pd.DataFrame(rows).sort_values("importance_abs_1sd", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Binary tag classifiers (shared by D_train_tag_predictors.py)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95, random_state=42
):
    """Bootstrap confidence interval for a scalar metric. Returns dict with 'mean', 'lower', 'upper'."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {"mean": np.nan, "lower": np.nan, "upper": np.nan}

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        try:
            score = metric_func(y_true_boot, y_pred_boot)
            if np.isfinite(score):
                bootstrap_scores.append(score)
        except:
            pass

    if len(bootstrap_scores) == 0:
        return {"mean": np.nan, "lower": np.nan, "upper": np.nan}

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(bootstrap_scores)),
        "lower": float(lower),
        "upper": float(upper),
    }
