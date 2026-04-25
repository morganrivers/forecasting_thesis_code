"""
Compares two sets of RF predictions on the validation set:

  A) "in-sample" -- pred_rf from best_model_predictions.csv, produced by
     A_overall_rating_fit_and_evaluate.py with USE_VAL_IN_TRAIN=True.  The model was trained on
     train+val combined, so val predictions are in-sample / overfit.

  B) "out-of-sample" -- predictions from an RF+ET model retrained HERE on
     train-only activities, using the same hyperparameters and the same
     recency correction, then applied to val.

The script loads everything from saved CSVs (all_features.csv,
best_model_predictions.csv, feature_names.json) so no re-running of the
full pipeline is needed.

Outputs a 4-panel figure saved next to this script:
  Panel 1: scatter of in-sample vs OOS predictions for val activities
  Panel 2: within-group pairwise ranking (both methods, vs random baseline)
  Panel 3: R^2 comparison
  Panel 4: MAE comparison

Usage (from repo root):
    python src/pipeline/D_overall_rating_generate_oos_predictions.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# -- paths --------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "data"
MODEL_DIR = DATA / "rating_model_outputs"

UTILS = REPO / "src" / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from ml_models import DISABLE_ET
from scoring_metrics import (
    mae as mae_metric,
)
from scoring_metrics import (
    within_group_pairwise_ordering_prob,
)

OUT_PLOT = Path(__file__).parent / "rf_val_insample_vs_oos.png"

# -- same hyperparameters as A_overall_rating_fit_and_evaluate.py ---------------------------
RF_PARAMS = {
    "n_estimators": 638,
    "max_depth": 14,
    "min_samples_split": 20,
    "min_samples_leaf": 20,
    "max_features": 0.488,
    "bootstrap": True,
    "max_samples": 0.86,
    "ccp_alpha": 1.26e-6,
    "random_state": 43,
    "n_jobs": -1,
}


# -- helpers -------------------------------------------------------------------


def compute_per_org_mode(ratings: pd.Series, orgs: pd.Series) -> pd.Series:
    """Return a Series of the training-set mode rating for each activity's org."""
    overall_mode = float(ratings.mode().iloc[0])
    mode_map = {}
    for org in orgs.unique():
        sub = ratings[orgs == org]
        mode_map[org] = float(sub.mode().iloc[0]) if len(sub) else overall_mode
    return orgs.map(mode_map).fillna(overall_mode)


def apply_recency_correction(
    pred: pd.Series,
    y: pd.Series,
    start_year: pd.Series,
    train_mask: pd.Index,
) -> pd.Series:
    """
    Fit residual ~ start_year on train set only; add the linear trend to pred
    everywhere (same logic as apply_start_year_trend_correction in A_overall_rating_fit_and_evaluate.py).
    """
    tr_idx = train_mask.intersection(pred.index).intersection(y.index)
    tr_idx = tr_idx[
        pred.loc[tr_idx].notna()
        & y.loc[tr_idx].notna()
        & start_year.loc[tr_idx].notna()
    ]
    years_tr = start_year.loc[tr_idx].astype(float)
    resid_tr = y.loc[tr_idx].astype(float) - pred.loc[tr_idx].astype(float)
    slope, intercept, r, p, _ = linregress(years_tr, resid_tr)
    print(
        f"  Recency correction: slope={slope:+.4f}/yr  intercept={intercept:.4f}  "
        f"r={r:.3f}  p={p:.4f}  n={len(tr_idx)}"
    )
    correction = intercept + slope * start_year.astype(float)
    return (pred + correction).clip(0.0, 5.0)


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


# -- load saved artefacts ------------------------------------------------------

print("Loading all_features.csv ...")
feat_df = pd.read_csv(MODEL_DIR / "all_features.csv", index_col="activity_id")
print(f"  {len(feat_df)} activities  x  {feat_df.shape[1]} columns")
print(f"  splits: {feat_df['split'].value_counts().to_dict()}")

print("\nLoading feature_names.json ...")
with open(MODEL_DIR / "feature_names.json") as f:
    feature_cols = json.load(f)
# keep only columns actually present
feature_cols = [c for c in feature_cols if c in feat_df.columns]
print(f"  {len(feature_cols)} feature columns found in CSV")

print("\nLoading best_model_predictions.csv ...")
bmp = pd.read_csv(DATA / "best_model_predictions.csv", index_col="activity_id")
print(f"  {len(bmp)} rows")

# -- split ---------------------------------------------------------------------

train_mask = feat_df[feat_df["split"] == "train"].index
val_mask = feat_df[feat_df["split"] == "val"].index
print(f"\nSplit sizes -- train: {len(train_mask)}  val: {len(val_mask)}")

y = feat_df["rating"].astype(float)
orgs = feat_df["reporting_orgs"].astype(str)
years = feat_df["start_year"].astype(float)

# -- A) in-sample pred_rf (USE_VAL_IN_TRAIN=True) -----------------------------

insample_pred = bmp["pred_rf"].reindex(val_mask)
print(f"\nIn-sample pred_rf: {insample_pred.notna().sum()} val activities have a value")

# -- B) out-of-sample: retrain on train-only -----------------------------------

print("\nComputing per-org mode from train-only data ...")
per_org_mode_train = compute_per_org_mode(y.loc[train_mask], orgs.loc[train_mask])
# Build a mapping: org -> mode, then apply to ALL activities
org_mode_map = {}
for org in orgs.loc[train_mask].unique():
    sub = y.loc[train_mask][orgs.loc[train_mask] == org]
    org_mode_map[org] = (
        float(sub.mode().iloc[0])
        if len(sub)
        else float(y.loc[train_mask].mode().iloc[0])
    )
overall_mode = float(y.loc[train_mask].mode().iloc[0])
per_org_mode_all = orgs.map(org_mode_map).fillna(overall_mode)

# Rating delta for training
rating_delta = y - per_org_mode_all

# Median-impute using train-only medians
X_all = feat_df[feature_cols].copy().astype(float)
train_medians = X_all.loc[train_mask].median()
X_imp = X_all.fillna(train_medians)

X_train_imp = X_imp.loc[train_mask]
y_delta_train = rating_delta.loc[train_mask]

print(f"Fitting RF on train-only ({len(train_mask)} activities) ...")
rf_oos = RandomForestRegressor(**RF_PARAMS)
rf_oos.fit(X_train_imp, y_delta_train)

print("Predicting on all activities ...")
if DISABLE_ET:
    delta_pred = rf_oos.predict(X_imp)
else:
    et_params = {**RF_PARAMS, "random_state": RF_PARAMS["random_state"] + 1}
    et_oos = ExtraTreesRegressor(**et_params)
    et_oos.fit(X_train_imp, y_delta_train)
    delta_pred = (rf_oos.predict(X_imp) + et_oos.predict(X_imp)) / 2.0
pred_raw = pd.Series(delta_pred, index=feat_df.index) + per_org_mode_all

print("Applying recency correction (fitted on train-only) ...")
pred_oos = apply_recency_correction(pred_raw, y, years, train_mask)

# -- align both sets to the val activities present in both ---------------------

common_val = val_mask.intersection(insample_pred.dropna().index).intersection(
    pred_oos.dropna().index
)
print(f"\nOverlapping val activities: {len(common_val)}")

y_val = y.loc[common_val].to_numpy()
pred_in = insample_pred.reindex(common_val).to_numpy()
pred_out = pred_oos.reindex(common_val).to_numpy()
orgs_val = orgs.reindex(common_val).to_numpy()
years_val = years.reindex(common_val).astype(str).to_numpy()

# within-group label = org + "_" + year
groups_val = np.array([f"{o}_{yr}" for o, yr in zip(orgs_val, years_val, strict=False)])

# -- metrics -------------------------------------------------------------------


def compute_metrics(y_true, y_pred, groups, label):
    wg = within_group_pairwise_ordering_prob(y_true, y_pred, groups)
    r2_val = r2(y_true, y_pred)
    mae_val = float(mae_metric(y_true, y_pred))
    print(f"\n{label}:")
    print(
        f"  Within-group pairwise: {wg['prob']:.4f}  (n_pairs={wg['n_pairs']}, n_groups={wg['n_groups']})"
    )
    print(f"  R^2:  {r2_val:.4f}")
    print(f"  MAE: {mae_val:.4f}")
    return {
        "wg_pairwise": wg["prob"],
        "r2": r2_val,
        "mae": mae_val,
        "n_pairs": wg["n_pairs"],
        "n_groups": wg["n_groups"],
    }


m_in = compute_metrics(y_val, pred_in, groups_val, "In-sample  (USE_VAL_IN_TRAIN=True)")
m_out = compute_metrics(y_val, pred_out, groups_val, "OOS        (train-only)")

# -- also compute on the FULL val set for OOS (insample may have fewer) --------
y_val_full = y.loc[val_mask].dropna()
pred_out_full = pred_oos.reindex(y_val_full.index)
orgs_full = orgs.reindex(y_val_full.index).to_numpy()
years_full = years.reindex(y_val_full.index).astype(str).to_numpy()
groups_full = np.array([f"{o}_{yr}" for o, yr in zip(orgs_full, years_full, strict=False)])
valid_full = ~np.isnan(pred_out_full.to_numpy())
m_out_full = compute_metrics(
    y_val_full.to_numpy()[valid_full],
    pred_out_full.to_numpy()[valid_full],
    groups_full[valid_full],
    "OOS (train-only, FULL val set)",
)

# -- plot ---------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    "RF Validation Predictions: In-sample (val in train) vs Out-of-sample (train-only)\n"
    f"n={len(common_val)} overlapping val activities",
    fontsize=14,
    fontweight="bold",
)

# -- Panel 1: scatter of predictions ------------------------------------------
ax = axes[0, 0]
sc = ax.scatter(
    pred_in,
    pred_out,
    c=y_val,
    cmap="RdYlGn",
    alpha=0.6,
    s=40,
    vmin=0,
    vmax=5,
    edgecolors="none",
)
lims = [
    min(pred_in.min(), pred_out.min()) - 0.1,
    max(pred_in.max(), pred_out.max()) + 0.1,
]
ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("In-sample pred_rf (val in train)", fontsize=11)
ax.set_ylabel("OOS pred_rf (train-only)", fontsize=11)
ax.set_title("Prediction scatter (colour = true rating)", fontsize=11)
fig.colorbar(sc, ax=ax, label="True rating")
ax.legend(fontsize=9)
corr = float(np.corrcoef(pred_in, pred_out)[0, 1])
ax.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax.transAxes, fontsize=10)
ax.grid(alpha=0.3)

# difference histogram
diff = pred_out - pred_in
ax2 = axes[0, 1]
ax2.hist(diff, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
ax2.axvline(0, color="k", lw=1.5, ls="--")
ax2.axvline(
    diff.mean(), color="tomato", lw=1.5, ls="-", label=f"mean={diff.mean():+.3f}"
)
ax2.set_xlabel("OOS pred - In-sample pred", fontsize=11)
ax2.set_ylabel("Count", fontsize=11)
ax2.set_title("Distribution of prediction differences", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# -- Panels 3 & 4: metric bars -------------------------------------------------
metric_pairs = [
    ("wg_pairwise", "Within-group pairwise ranking", [0.45, 0.65]),
    ("r2", "R^2", None),
    ("mae", "MAE", None),
]

bar_labels = ["In-sample\n(val in train)", "OOS\n(train-only)", "OOS\n(full val)"]
bar_colors = ["#d62728", "#1f77b4", "#2ca02c"]

for ax_idx, (key, title, ylim) in enumerate(metric_pairs[1:], start=0):
    ax = axes[1, ax_idx]
    vals = [m_in[key], m_out[key], m_out_full[key]]
    bars = ax.bar(bar_labels, vals, color=bar_colors, edgecolor="black", width=0.5)
    for bar, v in zip(bars, vals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(key.upper(), fontsize=10)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis="y", alpha=0.3)

# within-group pairwise as its own panel (replaces an axis)
# put it in row 1 col 0 but overwrite with the pairwise one
ax = axes[1, 0]  # re-use; already drawn above -- we'll redo it properly
ax.clear()
key, title = "wg_pairwise", "Within-group pairwise ranking"
vals = [m_in[key], m_out[key], m_out_full[key]]
bars = ax.bar(bar_labels, vals, color=bar_colors, edgecolor="black", width=0.5)
for bar, v in zip(bars, vals, strict=False):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{v:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax.axhline(0.5, color="gray", ls="--", lw=1.2, label="random baseline (0.5)")
ax.set_title(title, fontsize=12, fontweight="bold")
ax.set_ylabel("Prob. correct pair order", fontsize=10)
ax.set_ylim([0.4, 0.75])
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# mae panel
ax = axes[1, 1]
ax.clear()
key, title = "mae", "MAE"
vals = [m_in[key], m_out[key], m_out_full[key]]
bars = ax.bar(bar_labels, vals, color=bar_colors, edgecolor="black", width=0.5)
for bar, v in zip(bars, vals, strict=False):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{v:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
ax.set_title(title, fontsize=12, fontweight="bold")
ax.set_ylabel("MAE", fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUT_PLOT}")
plt.close(fig)

# -- save OOS predictions so F_llm_score_forecast_narratives.py can use them -----------
OOS_PRED_CSV = DATA / "rf_oos_val_predictions.csv"
oos_save = pred_oos.reindex(val_mask).rename("pred_rf_oos").reset_index()
oos_save.columns = ["activity_id", "pred_rf_oos"]
oos_save.to_csv(OOS_PRED_CSV, index=False)
print(
    f"\nOOS val predictions saved to {OOS_PRED_CSV}  ({oos_save['pred_rf_oos'].notna().sum()} rows)"
)

# -- summary -------------------------------------------------------------------
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Metric':<30} {'In-sample':>12} {'OOS (overlap)':>14} {'OOS (full val)':>15}")
print("-" * 65)
for key in ["wg_pairwise", "r2", "mae"]:
    print(f"{key:<30} {m_in[key]:>12.4f} {m_out[key]:>14.4f} {m_out_full[key]:>15.4f}")
print("=" * 65)
print(f"\nPrediction correlation (in-sample vs OOS): r = {corr:.4f}")
print(f"Mean difference (OOS - in-sample): {diff.mean():+.4f}  std={diff.std():.4f}")
print(f"Max abs difference: {np.abs(diff).max():.4f}")
