"""
In-sample variance explained for train+val using OLS (GLM) and Ridge.

Loads saved features from A_overall_rating_fit_and_evaluate.py (all_features.csv) and fits
both models on train+val combined, then prints:
  - R^2 and adjusted R^2 for each model
  - Per-feature semi-partial R^2: drop in R^2 when each feature is removed

Usage:
    python C_overall_rating_insample_r2.py [--all] [--train-only]

Arguments:
    --all         Train on train+val+test combined (no test-set evaluation reported).
    --train-only  Train on train set only, then report out-of-time val evaluation.
                  Default: train on train+val, then report out-of-time test evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from scoring_metrics import (
    adjusted_r2,
    within_group_pairwise_ordering_prob,
)

OUTPUT_DIR = Path("../../data/rating_model_outputs")
ALL_FEATURES_CSV = OUTPUT_DIR / "all_features.csv"


def fit_ols(X: np.ndarray, y: np.ndarray):
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const)
    return model.fit()


def fit_ridge(X: np.ndarray, y: np.ndarray, alphas=None):
    if alphas is None:
        alphas = np.logspace(-3, 4, 60)
    pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, fit_intercept=True))
    pipe.fit(X, y)
    alpha_chosen = pipe.named_steps["ridgecv"].alpha_
    print(f"Ridge RidgeCV best alpha: {alpha_chosen:.4g}")
    return pipe


def semi_partial_r2_ols(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    r2_full: float,
) -> pd.DataFrame:
    """Drop in R^2 when each feature is removed (semi-partial R^2) for OLS."""
    n, p = X.shape
    rows = []
    for i, name in enumerate(feature_names):
        mask = [j for j in range(p) if j != i]
        res = fit_ols(X[:, mask], y)
        drop = r2_full - float(res.rsquared)
        rows.append({"feature": name, "semi_partial_r2": drop})
    return (
        pd.DataFrame(rows)
        .sort_values("semi_partial_r2", ascending=False)
        .reset_index(drop=True)
    )


def ridge_coef_contribution(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    ridge_pipe,
) -> pd.DataFrame:
    """
    Per-feature contribution to R^2 for Ridge via beta_standardized x corr(x_j, y).

    For OLS this is exact (sums to R^2). For Ridge the betas are shrunk so the
    sum will be less than R^2, but the relative ordering is well-defined and
    avoids the negative-value problem of semi-partial R^2 under regularization.
    """
    scaler = ridge_pipe.named_steps["standardscaler"]
    ridge = ridge_pipe.named_steps["ridgecv"]

    beta_scaled = ridge.coef_.ravel()

    X_std = scaler.transform(X)
    corr_with_y = np.array(
        [float(np.corrcoef(X_std[:, j], y)[0, 1]) for j in range(X.shape[1])]
    )

    contributions = beta_scaled * corr_with_y
    rows = [
        {"feature": name, "coef_contribution": float(c)}
        for name, c in zip(feature_names, contributions, strict=False)
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values("coef_contribution", ascending=False)
        .reset_index(drop=True)
    )
    return df


def print_model_summary(label: str, r2: float, adj_r2: float, n: int, p: int):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  N = {n},  p = {p} features")
    print(f"  R^2          = {r2:.4f}")
    print(f"  Adjusted R^2 = {adj_r2:.4f}")


def print_semi_partial_table(df: pd.DataFrame, r2_full: float):
    print(f"\n  {'Feature':<40} {'Semi-partial R^2':>16} {'% of total R^2':>14}")
    print(f"  {'-'*40} {'-'*16} {'-'*14}")
    for _, row in df.iterrows():
        pct = (row["semi_partial_r2"] / r2_full * 100) if r2_full != 0 else np.nan
        print(f"  {row['feature']:<40} {row['semi_partial_r2']:>16.4f} {pct:>13.1f}%")
    print(f"  {'-'*40} {'-'*16} {'-'*14}")
    total_spr2 = df["semi_partial_r2"].sum()
    print(f"  {'Sum of semi-partial R^2':<40} {total_spr2:>16.4f}")
    print(f"  {'Full model R^2':<40} {r2_full:>16.4f}")


def print_ridge_contribution_table(df: pd.DataFrame, r2_full: float):
    total = df["coef_contribution"].sum()
    print(f"\n  {'Feature':<40} {'beta*corr(x,y)':>12} {'% of beta*corr sum':>16}")
    print(f"  {'-'*40} {'-'*12} {'-'*16}")
    for _, row in df.iterrows():
        pct = (row["coef_contribution"] / total * 100) if total != 0 else np.nan
        print(f"  {row['feature']:<40} {row['coef_contribution']:>12.4f} {pct:>15.1f}%")
    print(f"  {'-'*40} {'-'*12} {'-'*16}")
    print(
        f"  {'Sum (beta*corr decomposition)':<40} {total:>12.4f}  (~= R^2 for OLS; < R^2 for Ridge)"
    )
    print(f"  {'Full model R^2':<40} {r2_full:>12.4f}")


def compute_org_modes(
    df_fit: pd.DataFrame, org_cols: list, y_col: str = "rating"
) -> tuple:
    y = df_fit[y_col].astype(float)
    overall_mode = float(y.mode().iloc[0])
    org_modes = {}
    for c in org_cols:
        sub = y[df_fit[c].fillna(0).astype(int) == 1]
        org_modes[c] = float(sub.mode().iloc[0]) if len(sub) else overall_mode
    return overall_mode, org_modes


def apply_org_modes(
    df_split: pd.DataFrame, overall_mode: float, org_modes: dict, org_cols: list
) -> pd.Series:
    pom = pd.Series(overall_mode, index=df_split.index, dtype=float)
    for c in org_cols:
        pom[df_split[c].fillna(0).astype(int) == 1] = org_modes[c]
    return pom


def main():
    use_all = "--all" in sys.argv
    train_only = "--train-only" in sys.argv

    if use_all and train_only:
        raise SystemExit("Error: --all and --train-only are mutually exclusive.")

    print(f"Loading {ALL_FEATURES_CSV}")
    df = pd.read_csv(ALL_FEATURES_CSV, index_col="activity_id")
    print(
        f"Loaded {len(df)} rows. Splits present: {df['split'].value_counts().to_dict()}"
    )

    required_cols = ["split", "rating", "reporting_orgs", "start_year"]
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        raise KeyError(
            f"all_features.csv is missing required columns {missing_req}. "
            "Re-run A_overall_rating_fit_and_evaluate.py to regenerate it."
        )

    if use_all:
        splits_to_use = {"train", "val", "test"}
        print("Mode: --all  (train+val+test, no test-set evaluation)")
    elif train_only:
        splits_to_use = {"train"}
        print("Mode: --train-only  (train only, with out-of-time val evaluation)")
    else:
        splits_to_use = {"train", "val"}
        print("Mode: default (train+val, with out-of-time test evaluation)")

    df_sub = df[df["split"].isin(splits_to_use)].copy()
    print(f"Using splits {sorted(splits_to_use)}: {len(df_sub)} rows")

    org_cols = [c for c in df_sub.columns if c.startswith("rep_org_")]

    # Org modes computed on the same set we fit on
    overall_mode, org_modes = compute_org_modes(
        df_sub.dropna(subset=["rating"]), org_cols
    )
    per_org_mode = apply_org_modes(df_sub, overall_mode, org_modes, org_cols)

    print(f"Per-org modes (from fitting set): overall={overall_mode}")
    for c, m in org_modes.items():
        print(f"  {c}: {m}")

    y_raw = df_sub["rating"].astype(float)
    y_delta_series = y_raw - per_org_mode
    meta_cols = {"split", "rating", "reporting_orgs", "start_year"}
    feature_names = [c for c in df_sub.columns if c not in meta_cols]

    valid = y_raw.notna()
    df_sub = df_sub[valid]
    y_delta_series = y_delta_series[valid]
    print(f"After dropping missing ratings: {len(df_sub)} rows")
    print(
        f"Target: rating_delta  mean={y_delta_series.mean():.3f}  std={y_delta_series.std():.3f}"
    )

    X_raw = df_sub[feature_names].copy()

    medians = X_raw.median(numeric_only=True)
    all_nan_cols = X_raw.columns[X_raw.isna().all()].tolist()
    if all_nan_cols:
        print(f"Dropping all-NaN columns: {all_nan_cols}")
        feature_names = [f for f in feature_names if f not in all_nan_cols]
        X_raw = X_raw[feature_names]

    X_imp = X_raw.fillna(medians)
    X = X_imp.to_numpy(dtype=float)
    y = y_delta_series.to_numpy(dtype=float)

    n, p = X.shape
    print(f"Feature matrix: {n} rows x {p} features")

    # ---------------------------------------------------------------------------
    # OLS
    # ---------------------------------------------------------------------------
    print("\nFitting OLS...")
    ols_result = fit_ols(X, y)
    r2_ols = float(ols_result.rsquared)
    adj_r2_ols = float(ols_result.rsquared_adj)

    print_model_summary("OLS (GLM Gaussian)", r2_ols, adj_r2_ols, n, p)

    print("\nComputing semi-partial R^2 for OLS (refitting without each feature)...")
    spr2_ols = semi_partial_r2_ols(X, y, feature_names, r2_ols)
    print_semi_partial_table(spr2_ols, r2_ols)

    # ---------------------------------------------------------------------------
    # Ridge
    # ---------------------------------------------------------------------------
    print("\n\nFitting Ridge...")
    ridge_pipe = fit_ridge(X, y)
    y_hat_ridge = ridge_pipe.predict(X)
    r2_ridge = float(r2_score(y, y_hat_ridge))
    adj_r2_ridge = adjusted_r2(r2_ridge, n, p)

    print_model_summary("Ridge (RidgeCV, StandardScaler)", r2_ridge, adj_r2_ridge, n, p)

    print("\nPer-feature contribution (beta_std x corr(x,y)) -- Ridge:")
    print("  Note: unlike OLS semi-partial R^2, this doesn't refit the model.")
    print("  Negative values mean that feature pushes predictions away from y.")
    ridge_contrib = ridge_coef_contribution(X, y, feature_names, ridge_pipe)
    print_ridge_contribution_table(ridge_contrib, r2_ridge)

    # ---------------------------------------------------------------------------
    # Side-by-side comparison
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*60}")
    print("  Summary comparison")
    print(f"{'='*60}")
    print(f"  {'Model':<35} {'R^2':>8} {'Adj R^2':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*10}")
    print(f"  {'OLS (GLM Gaussian)':<35} {r2_ols:>8.4f} {adj_r2_ols:>10.4f}")
    print(f"  {'Ridge (RidgeCV)':<35} {r2_ridge:>8.4f} {adj_r2_ridge:>10.4f}")

    comparison = (
        spr2_ols[["feature", "semi_partial_r2"]]
        .merge(
            ridge_contrib[["feature", "coef_contribution"]],
            on="feature",
        )
        .sort_values("semi_partial_r2", ascending=False)
    )

    print(f"\n  {'Feature':<40} {'OLS spr2':>10} {'Ridge beta*corr':>14}")
    print(f"  {'-'*40} {'-'*10} {'-'*14}")
    for _, row in comparison.iterrows():
        print(
            f"  {row['feature']:<40} {row['semi_partial_r2']:>10.4f} {row['coef_contribution']:>14.4f}"
        )

    if use_all:
        return

    # ---------------------------------------------------------------------------
    # Out-of-time evaluation:
    #   --train-only: train -> val
    #   default:      train+val -> test
    # ---------------------------------------------------------------------------
    if train_only:
        fit_splits = ["train"]
        eval_split = "val"
        eval_label = "val"
        print(f"\n\n{'='*60}")
        print("  OLS val-set evaluation (train -> val, out-of-time)")
        print(f"{'='*60}")
    else:
        fit_splits = ["train", "val"]
        eval_split = "test"
        eval_label = "test"
        print(f"\n\n{'='*60}")
        print("  OLS test-set evaluation (train+val -> test, out-of-time)")
        print(f"{'='*60}")

    df_fit = df[df["split"].isin(fit_splits)].copy()
    df_eval = df[df["split"] == eval_split].copy()

    df_fit = df_fit[df_fit["rating"].notna()]
    df_eval = df_eval[df_eval["rating"].notna()]

    overall_mode_tv, org_modes_tv = compute_org_modes(df_fit, org_cols)
    pom_fit = apply_org_modes(df_fit, overall_mode_tv, org_modes_tv, org_cols)
    pom_eval = apply_org_modes(df_eval, overall_mode_tv, org_modes_tv, org_cols)

    y_delta_fit = (df_fit["rating"].astype(float) - pom_fit).to_numpy(dtype=float)
    y_true_eval = df_eval["rating"].astype(float).to_numpy(dtype=float)

    X_fit_raw = df_fit[feature_names].copy()
    X_eval_raw = df_eval[feature_names].copy()
    fit_medians = X_fit_raw.median(numeric_only=True)
    X_fit_imp = X_fit_raw.fillna(fit_medians).to_numpy(dtype=float)
    X_eval_imp = X_eval_raw.fillna(fit_medians).to_numpy(dtype=float)

    X_fit_const = sm.add_constant(X_fit_imp, has_constant="add")
    X_eval_const = sm.add_constant(X_eval_imp, has_constant="add")
    ols_fit = sm.OLS(y_delta_fit, X_fit_const).fit()

    y_pred_delta_eval = ols_fit.predict(X_eval_const)
    y_pred_eval = y_pred_delta_eval + pom_eval.to_numpy(dtype=float)

    r2_eval = float(r2_score(y_true_eval, y_pred_eval))

    groups_eval = (
        df_eval["reporting_orgs"].astype(str) + "_" + df_eval["start_year"].astype(str)
    ).to_numpy()
    _r_wg = within_group_pairwise_ordering_prob(y_true_eval, y_pred_eval, groups_eval)
    wg_pairwise, wg_n_pairs, wg_n_groups = (
        _r_wg["prob"],
        _r_wg["n_pairs"],
        _r_wg["n_groups"],
    )

    n_eval = len(y_true_eval)
    print(f"  N {eval_label} = {n_eval}")
    print(f"  R^2 ({eval_label})                                    = {r2_eval:.4f}")
    print(
        f"  WG pairwise (orgxyear, {eval_label}, excl. ties)     = {wg_pairwise:.4f}  ({wg_n_pairs} pairs, {wg_n_groups} groups)"
    )


if __name__ == "__main__":
    main()
