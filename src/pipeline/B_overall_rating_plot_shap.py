"""
SHAP beeswarm plot (top 10 features) using the saved TreeExplainer and
train_features.csv. Saves feature_importances_plot.png.

Usage:
    python B_overall_rating_plot_shap.py
"""

from pathlib import Path
import sys
import json
import csv
import pickle
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.overall_rating_feature_labels import get_display_name

MODEL_PKL = REPO_ROOT / "data" / "rating_model_outputs" / "model.pkl"
FEAT_CSV = REPO_ROOT / "data" / "rating_model_outputs" / "train_features.csv"
MED_JSON = REPO_ROOT / "data" / "rating_model_outputs" / "train_medians.json"
FEAT_NAMES_J = REPO_ROOT / "data" / "rating_model_outputs" / "feature_names.json"
OUT_PATH = Path(__file__).resolve().parent / "feature_importances_plot.png"

# ---------------------------------------------------------------------------
# Load feature names, explainer, and features
# ---------------------------------------------------------------------------
with open(FEAT_NAMES_J) as f:
    feature_names = json.load(f)

with open(MED_JSON) as f:
    medians = json.load(f)

X_rows = []
with open(FEAT_CSV) as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    for row in reader:
        X_rows.append([float(row[c]) if row[c] != "" else np.nan for c in cols])

X = np.array(X_rows)
for j, col in enumerate(cols):
    mask = np.isnan(X[:, j])
    if mask.any():
        X[mask, j] = medians.get(col, 0.0)

X_df = pd.DataFrame(X, columns=feature_names)

with open(MODEL_PKL, "rb") as f:
    rf_model = pickle.load(f)
explainer = shap.TreeExplainer(rf_model)

# ---------------------------------------------------------------------------
# Compute SHAP values
# ---------------------------------------------------------------------------
print(f"Computing SHAP values for {len(X_df)} training samples...")
shap_values = explainer.shap_values(X_df)  # (n_train, n_features)

# ---------------------------------------------------------------------------
# Select top 10 by mean |SHAP|
# ---------------------------------------------------------------------------
mean_abs = np.abs(shap_values).mean(axis=0)
top10_idx = np.argsort(mean_abs)[::-1][:15]
top10_feats = [feature_names[i] for i in top10_idx]

shap_top10 = shap_values[:, top10_idx]
X_top10 = X_df.iloc[:, top10_idx].copy()
display_names = [get_display_name(f) for f in top10_feats]
X_top10.columns = display_names

print("Top 15 features:")
for feat, val, disp in zip(top10_feats, mean_abs[top10_idx], display_names):
    print(f"  {feat:<50}  mean|SHAP|={val:.4f}  ->  '{disp}'")

# ---------------------------------------------------------------------------
# Beeswarm
# ---------------------------------------------------------------------------
print("Plotting beeswarm...")
fig = plt.figure(figsize=(9, 7.5))
shap.summary_plot(
    shap_top10,
    X_top10,
    show=False,
    max_display=15,
    plot_size=None,
)
plt.title(
    "SHAP feature importance -- top 15 features, RF (training set)",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")

# ---------------------------------------------------------------------------
# ExtraTrees SHAP (if extra_model.pkl exists)
# ---------------------------------------------------------------------------
EXTRA_PKL = REPO_ROOT / "data" / "rating_model_outputs" / "extra_model.pkl"
OUT_PATH_ET = Path(__file__).resolve().parent / "feature_importances_plot_et.png"

if EXTRA_PKL.exists():
    print("Computing ExtraTrees SHAP values...")
    with open(EXTRA_PKL, "rb") as f:
        et_model = pickle.load(f)
    et_explainer = shap.TreeExplainer(et_model)
    et_shap_values = et_explainer.shap_values(X_df)

    et_mean_abs = np.abs(et_shap_values).mean(axis=0)
    et_top_idx = np.argsort(et_mean_abs)[::-1][:15]
    et_top_feats = [feature_names[i] for i in et_top_idx]

    et_shap_top = et_shap_values[:, et_top_idx]
    et_X_top = X_df.iloc[:, et_top_idx].copy()
    et_display_names = [get_display_name(f) for f in et_top_feats]
    et_X_top.columns = et_display_names

    print("Top 15 features (ExtraTrees):")
    for feat, val, disp in zip(et_top_feats, et_mean_abs[et_top_idx], et_display_names):
        print(f"  {feat:<50}  mean|SHAP|={val:.4f}  ->  '{disp}'")

    fig2 = plt.figure(figsize=(9, 7.5))
    shap.summary_plot(et_shap_top, et_X_top, show=False, max_display=15, plot_size=None)
    plt.title(
        "SHAP feature importance -- top 15 features, ExtraTrees (training set)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    fig2.savefig(OUT_PATH_ET, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PATH_ET}")

    # Averaged RF + ET SHAP
    print("Computing averaged RF+ET SHAP values...")
    avg_shap = (shap_values + et_shap_values) / 2.0
    avg_mean_abs = np.abs(avg_shap).mean(axis=0)
    avg_top_idx = np.argsort(avg_mean_abs)[::-1][:15]
    avg_top_feats = [feature_names[i] for i in avg_top_idx]

    avg_shap_top = avg_shap[:, avg_top_idx]
    avg_X_top = X_df.iloc[:, avg_top_idx].copy()
    avg_display_names = [get_display_name(f) for f in avg_top_feats]
    avg_X_top.columns = avg_display_names

    print("Top 15 features (RF+ET average):")
    for feat, val, disp in zip(
        avg_top_feats, avg_mean_abs[avg_top_idx], avg_display_names
    ):
        print(f"  {feat:<50}  mean|SHAP|={val:.4f}  ->  '{disp}'")

    OUT_PATH_AVG = (
        Path(__file__).resolve().parent / "feature_importances_plot_ensemble.png"
    )
    fig3 = plt.figure(figsize=(9, 7.5))
    shap.summary_plot(
        avg_shap_top, avg_X_top, show=False, max_display=15, plot_size=None
    )
    plt.title(
        "SHAP feature importance -- top 15 features, RF+ET ensemble (training set)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    fig3.savefig(OUT_PATH_AVG, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PATH_AVG}")
else:
    print("extra_model.pkl not found -- skipping ET SHAP plot")
