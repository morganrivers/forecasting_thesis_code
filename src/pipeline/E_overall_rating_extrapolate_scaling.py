"""
Scaling extrapolation and diagnostic analyses for the overall-rating model.

Methods evaluated:
  per_org_mode             -- Mode of reporting-org score baseline
  ridge_baseline_risk_org  -- Ridge Baseline (risks + org only)
  pred_rf                  -- RF+ET with custom tuned params (no LLM, no recency)
  pred_rf_llm_modded       -- RF+ET + LLM Forecast + recency  (best model)

Data for most analyses loaded from saved files (no model re-running):
  data/best_model_predictions.csv  -- all predictions + metadata columns
  data/train_val_test_ids.csv      -- split assignments per activity_id

Analyses that rebuild the full feature matrix are slow (~3-5 min).

Usage:
  python E_overall_rating_extrapolate_scaling.py --lc-kfold   # temporal k-fold CV
  python E_overall_rating_extrapolate_scaling.py --lc-pop     # learning curve
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# Path setup -- add utils and C_forecast_outcomes so we can import from there.
# Both src/D_data_analysis/ and src/C_forecast_outcomes/ sit at the same depth
# from the repo root, so relative paths like ../../data/ resolve identically.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # forecasting_iati/
UTILS_DIR = REPO_ROOT / "src" / "utils"
PIPELINE_DIR = REPO_ROOT / "src" / "pipeline"
for _p in [str(UTILS_DIR), str(PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scoring_metrics import (
    rmse,
    mae,
    r2 as r2_metric,
    true_hit_accuracy,
    side_accuracy,
    spearman_correlation,
    pairwise_ordering_prob_excl_ties,
    within_group_pairwise_ordering_prob,
)
from ml_models import (
    run_random_forest_median_impute_noclip,
    run_ridge_glm_median_impute_noclip,
    bootstrap_ci,
    apply_start_year_trend_correction,
)
from feature_engineering import (
    load_grades,
    load_is_completed,
    load_ratings,
    load_activity_scope,
    load_gdp_percap,
    load_implementing_org_type,
    load_world_bank_indicators,
    add_similarity_features,
    pick_start_date,
    add_dates_to_dataframe,
    restrict_to_reporting_orgs_exact,
    load_targets_context_maps_features,
    add_enhanced_uncertainty_features,
    data_sector_clusters,
)
from data_loan_disbursement import load_loan_or_disbursement
from llm_load_predictions import load_predictions_from_jsonl, get_llm_prediction_configs
from A_overall_rating_fit_and_evaluate import (
    KEEP_REPORTING_ORGS,
    NUM_ORGS_KEEP,
    LATEST_TRAIN_POINT,
    LATEST_VALIDATION_POINT,
    TOO_LATE_CUTOFF,
    split_latest_by_date_with_cutoff,
    add_per_org_mode_baseline,
    add_rf_llm_residual_corrector,
    load_llm_planned_expenditure,
    load_llm_planned_duration,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_DIR = REPO_ROOT / "data"
PREDICTIONS_CSV = DATA_DIR / "best_model_predictions.csv"
SPLITS_CSV = DATA_DIR / "train_val_test_ids.csv"
INFO_CSV = DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "generalizability_plots"

# RF hyperparameters -- must match A_overall_rating_fit_and_evaluate.py exactly
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

WGI_COLS = [
    "wgi_control_of_corruption_est",
    "wgi_political_stability_est",
    "wgi_government_effectiveness_est",
    "wgi_regulatory_quality_est",
    "wgi_rule_of_law_est",
]

METHODS = [
    ("Mode of reporting-org score baseline", "per_org_mode"),
    ("Ridge Baseline (risks + org only)", "ridge_baseline_risk_org"),
    ("RF+ET only (custom params)", "pred_rf"),
    ("RF+ET + LLM Forecast + recency", "pred_rf_llm_modded"),
]

SIDE_THRESHOLD = 3.5
NUM_RANDOM_TRAINING_SAMPLES_TO_AVERAGE = 5

_METHOD_SHORT = {
    "per_org_mode": "mode",
    "ridge_baseline_risk_org": "ridge",
    "pred_rf": "rf_only",
    "pred_rf_llm_modded": "rf+llm",
}

# Short names for KEEP_REPORTING_ORGS used in LOOO labels
_ORG_SHORT = {
    "UK - Foreign, Commonwealth Development Office (FCDO)": "FCDO",
    "Asian Development Bank": "ADB",
    "World Bank": "World Bank",
    "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); "
    "Federal Ministry for Economic Cooperation and Development (BMZ)": "BMZ",
}


# _excl_ties_pairwise_same_year removed -- use within_group_pairwise_ordering_prob
# from scoring_metrics (canonical implementation used across all files)


def analysis_17_learning_curve(
    data: pd.DataFrame,
    y: pd.Series,
    train_idx,
    val_idx,
    feature_cols: list,
    train_pool_idx=None,
    fixed_eval_idx=None,
    set_label: str = "val",
    r2_only: bool = False,
) -> None:
    """
    Learning curve: randomly drop 0%, 10%, 20%, 30%, 40%, 50% of training
    data and evaluate RF+ET+LLM on the full val set.  Fits a log model to
    project expected R^2 and POP if 5x more training data were available.
    Saves plot to this directory.

    train_pool_idx: if provided, used as the training pool (default: train_idx).
    fixed_eval_idx: if provided, used as the evaluation set (default: val_idx).
    set_label: label used in plot titles and output filenames ("val" or "test").
    r2_only: if True, produce only the single R^2 panel figure and skip the
             1/RMSE panel and the per-org POP figure.
    """
    effective_pool = train_pool_idx if train_pool_idx is not None else train_idx
    effective_eval = fixed_eval_idx if fixed_eval_idx is not None else val_idx
    print("\n" + "=" * 80)
    print("ANALYSIS 17: LEARNING CURVE (best method: RF+ET+LLM)")
    print("Drop fractions: 0% 10% 20% 30% 40% 50% 60% 70% 80% 90%")
    print("=" * 80)

    # All methods to track -- BEST is the ensemble+LLM+recency
    CURVE_METHODS = [
        ("per_org_mode", "Mode baseline", "grey", "--", 1.2),
        ("ridge_baseline_risk_org", "Ridge baseline", "orange", "-", 1.2),
        ("pred_rf", "RF+ET", "steelblue", "-", 1.2),
        ("pred_rf_llm_modded", "RF+ET+LLM+recency", "royalblue", "-", 1.5),
        ("ridge_plus_rf_llm_modded", "Ensemble+LLM+recency", "darkgreen", "-", 2.0),
    ]
    BEST_COL = "ridge_plus_rf_llm_modded"

    rng = np.random.default_rng(42)
    drop_fracs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_full = len(effective_pool)

    # rows: list of {drop_pct, n_train, <col>_R2, <col>_POP, ...}
    rows = []
    for frac in drop_fracs:
        n_keep = int(round(n_full * (1.0 - frac)))
        print(
            f"\n  Drop {int(frac*100):>2}%  ->  train n={n_keep}  (of {n_full})  "
            f"averaging {NUM_RANDOM_TRAINING_SAMPLES_TO_AVERAGE} random samples..."
        )

        yt = y.loc[effective_eval].astype(float).to_numpy()
        sample_metrics = {
            col: {"R2": [], "RMSE": [], "WG_POP": []} for col, *_ in CURVE_METHODS
        }

        # Same grouping as A_overall_rating_fit_and_evaluate.py: reporting_org + "_" + start_year
        all_groups = (
            data.loc[effective_eval, "reporting_orgs"].astype(str)
            + "_"
            + data.loc[effective_eval, "start_date"].dt.year.astype(str)
        ).to_numpy()

        for sample_i in range(NUM_RANDOM_TRAINING_SAMPLES_TO_AVERAGE):
            chosen = rng.choice(n_full, size=n_keep, replace=False)
            tr_idx = effective_pool[np.sort(chosen)]
            d = _train_and_predict_fold(data, y, feature_cols, tr_idx, effective_eval)
            for col, *_ in CURVE_METHODS:
                if col not in d.columns:
                    continue
                m = _metrics(yt, d.loc[effective_eval, col].astype(float).to_numpy())
                sample_metrics[col]["R2"].append(m["R2"])
                sample_metrics[col]["RMSE"].append(m["RMSE"])
                yp = d.loc[effective_eval, col].astype(float).to_numpy()
                pop = within_group_pairwise_ordering_prob(yt, yp, all_groups)["prob"]
                sample_metrics[col]["WG_POP"].append(pop)
            best_r2 = np.mean(sample_metrics[BEST_COL]["R2"])
            best_rmse = np.mean(sample_metrics[BEST_COL]["RMSE"])
            print(
                f"    sample {sample_i+1}/{NUM_RANDOM_TRAINING_SAMPLES_TO_AVERAGE}: "
                f"BEST R^2={best_r2:.4f}  RMSE={best_rmse:.4f}"
            )

        row = {"drop_pct": int(frac * 100), "n_train": n_keep}
        for col, *_ in CURVE_METHODS:
            vals = sample_metrics[col]
            row[f"{col}_R2"] = (
                float(np.mean(vals["R2"])) if vals["R2"] else float("nan")
            )
            row[f"{col}_RMSE"] = (
                float(np.mean(vals["RMSE"])) if vals["RMSE"] else float("nan")
            )
            row[f"{col}_WG_POP"] = (
                float(np.mean(vals["WG_POP"])) if vals["WG_POP"] else float("nan")
            )
        rows.append(row)
        print(
            f"    -> BEST mean R^2={row[f'{BEST_COL}_R2']:.4f}  "
            f"RMSE={row[f'{BEST_COL}_RMSE']:.4f}"
        )

    df = pd.DataFrame(rows)
    ns = df["n_train"].to_numpy(dtype=float)
    logn = np.log(ns)

    def _fit_log(logn_arr, y_arr):
        A = np.column_stack([np.ones_like(logn_arr), logn_arr])
        a, b = np.linalg.lstsq(A, y_arr, rcond=None)[0]
        return a, b

    # Log fit only for BEST method (on R^2, 1/RMSE, and mean POP)
    _fit_col = "pred_rf_llm_modded" if r2_only else BEST_COL
    best_r2s = df[f"{_fit_col}_R2"].to_numpy(dtype=float)
    best_rmses = df[f"{_fit_col}_RMSE"].to_numpy(dtype=float)
    best_inv_rmse = 1.0 / best_rmses
    a_r2, b_r2 = _fit_log(logn, best_r2s)
    a_inv, b_inv = _fit_log(logn, best_inv_rmse)

    # POP fit uses pred_rf (RF+ET + year correction, no LLM)
    # POP fit uses pred_rf with the same within_group_pairwise_ordering_prob
    # over all org-year groups as A_overall_rating_fit_and_evaluate.py (weighted by pairs, not avg of orgs)
    _pop_col = "pred_rf"
    mean_pop = df[f"{_pop_col}_WG_POP"].to_numpy(dtype=float)
    a_pop, b_pop = _fit_log(logn, mean_pop)

    n_proj = 5 * n_full
    r2_proj = a_r2 + b_r2 * np.log(n_proj)
    inv_proj = a_inv + b_inv * np.log(n_proj)
    pop_proj = a_pop + b_pop * np.log(n_proj)

    print(f"\n{'='*60}")
    print(f"Log model fit on BEST ({BEST_COL}):  metric = a + b·log(n)")
    print(f"  R^2      fit:  a={a_r2:+.4f}  b={b_r2:+.4f}")
    print(f"  1/RMSE  fit:  a={a_inv:+.4f}  b={b_inv:+.4f}")
    print(f"\nProjection to 5x training data (n={n_proj}):")
    print(
        f"  R^2     at n_full={n_full}: {a_r2  + b_r2 *np.log(n_full):.4f}  ->  "
        f"projected {r2_proj:.4f}   (Delta={r2_proj -(a_r2 +b_r2 *np.log(n_full)):+.4f})"
    )
    print(
        f"  1/RMSE at n_full={n_full}: {a_inv + b_inv*np.log(n_full):.4f}  ->  "
        f"projected {inv_proj:.4f}  (Delta={inv_proj-(a_inv+b_inv*np.log(n_full)):+.4f})"
    )
    pop_at_n_full_fitted = float(a_pop + b_pop * np.log(n_full))
    print(f"  WG-POP fit:   a={a_pop:+.4f}  b={b_pop:+.4f}")
    print(
        f"  pop_proj_5x at n_full={n_full}: {pop_at_n_full_fitted:.4f}  ->  "
        f"pop_proj_5x = {float(pop_proj):.4f}  (Delta={float(pop_proj)-pop_at_n_full_fitted:+.4f})"
    )
    print(f"\n  Caveat: 5x projection assumes same domain distribution.")

    BASE_FS = 18
    ns_curve = np.linspace(ns.min() * 0.5, n_proj, 300)
    r2_curve = a_r2 + b_r2 * np.log(ns_curve)
    inv_curve = a_inv + b_inv * np.log(ns_curve)

    _lc_suffix = "" if set_label == "val" else f"_{set_label}"

    _lc_csv = Path(__file__).resolve().parent / f"learning_curve_data{_lc_suffix}.csv"
    df.to_csv(_lc_csv, index=False)
    print(f"Learning curve data saved -> {_lc_csv}")

    # -- Plot 1: R^2 learning curve (single panel) ----------------------------
    _SKIP_COLS = (
        {"pred_rf", "ridge_baseline_risk_org", "ridge_plus_rf_llm_modded"}
        if r2_only
        else set()
    )
    _RENAME = {"pred_rf_llm_modded": "Combined Model"} if r2_only else {}
    fig1, ax_r2 = plt.subplots(1, 1, figsize=(10, 6))
    for col, label, color, ls, lw in CURVE_METHODS:
        if col in _SKIP_COLS:
            continue
        label = _RENAME.get(col, label)
        raw = df[f"{col}_R2"].to_numpy(dtype=float)
        ax_r2.plot(
            ns,
            raw,
            color=color,
            linestyle=ls,
            linewidth=lw,
            marker="o",
            markersize=5,
            label=label,
            zorder=4 if col == BEST_COL else 3,
        )
    ax_r2.plot(
        ns_curve,
        r2_curve,
        color="darkgreen",
        linestyle=":",
        linewidth=1.2,
        label="log fit (Combined Model)" if r2_only else "log fit (BEST)",
        zorder=2,
    )
    ax_r2.axvline(
        n_full, color="grey", linestyle="--", linewidth=1, label=f"current n={n_full}"
    )
    if not r2_only:
        ax_r2.axvline(
            n_proj, color="tomato", linestyle="--", linewidth=1, label=f"5x n={n_proj}"
        )
        ax_r2.scatter(
            [n_proj],
            [r2_proj],
            color="tomato",
            zorder=6,
            s=100,
            marker="*",
            label=f"projected={r2_proj:.3f}",
        )
    if r2_only:
        # -- Current-performance arrow annotation --------------------------
        _r2_at_full = float(a_r2 + b_r2 * np.log(n_full))
        ax_r2.annotate(
            f"Current performance\n(R^2={_r2_at_full:.3f}, N={n_full})",
            xy=(n_full, _r2_at_full),
            xytext=(n_full + 500, _r2_at_full + 0.01),
            fontsize=BASE_FS * 1.08,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="grey", lw=1.5),
            color="grey",
        )
        # -- N=5000 projected star + label ---------------------------------
        _n_iati = 5000
        _r2_iati = float(a_r2 + b_r2 * np.log(_n_iati))
        ax_r2.scatter(
            [_n_iati], [_r2_iati], color="tomato", zorder=6, s=120, marker="*"
        )
        ax_r2.annotate(
            f"Estimated performance using\nall available IATI data\n(N={_n_iati}, R^2={_r2_iati:.3f})",
            xy=(_n_iati, _r2_iati),
            xytext=(_n_iati - 2400, _r2_iati - 0.02),
            fontsize=BASE_FS * 1.08,
            ha="left",
            va="top",
            arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5),
            color="tomato",
        )
    ax_r2.set_xlabel("Training set size (n)", fontsize=BASE_FS)
    ax_r2.set_ylabel("R^2", fontsize=BASE_FS)
    ax_r2.set_title(f"Learning curve -- R^2  ({set_label} set)", fontsize=BASE_FS)
    ax_r2.legend(fontsize=BASE_FS * 0.72, loc="lower right")
    ax_r2.tick_params(axis="both", labelsize=BASE_FS * 0.82)
    ax_r2.grid(True, alpha=0.3)
    fig1.tight_layout()
    out1 = Path(__file__).resolve().parent / f"learning_curve{_lc_suffix}.png"
    fig1.savefig(out1, dpi=120)
    plt.close(fig1)
    print(f"\nPlot 1 saved -> {out1}")

    # -- Plot 2: single-panel ranking skill -- WB and BMZ on same axes --------
    # Use actual measured value at frac=0 (all training data), not the fitted curve value
    pop_at_n_full = float(mean_pop[0])
    pop_curve = a_pop + b_pop * np.log(ns_curve)

    fig2, ax_pop = plt.subplots(1, 1, figsize=(10, 6))

    _ridge_col = "ridge_baseline_risk_org"
    ridge_wg = df[f"{_ridge_col}_WG_POP"].to_numpy(dtype=float)
    best_wg = df[f"{_pop_col}_WG_POP"].to_numpy(dtype=float)
    ax_pop.plot(
        ns,
        ridge_wg,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        marker="o",
        markersize=5,
        label="Ridge baseline (all orgs)",
        zorder=3,
    )
    ax_pop.plot(
        ns,
        best_wg,
        color="royalblue",
        linestyle="-",
        linewidth=2.0,
        marker="o",
        markersize=6,
        label="RF+ET + year corr (all orgs)",
        zorder=4,
    )

    ax_pop.plot(
        ns_curve,
        pop_curve,
        color="grey",
        linestyle=":",
        linewidth=1.2,
        label="log fit (WG pairwise)",
        zorder=2,
    )
    ax_pop.axhline(
        0.5, color="black", linestyle=":", linewidth=0.8, label="chance (0.5)"
    )
    ax_pop.axvline(
        n_full, color="grey", linestyle="--", linewidth=1, label=f"current n={n_full}"
    )

    # -- Current-performance arrow annotation ------------------------------
    _pop_at_full = pop_at_n_full
    ax_pop.annotate(
        f"Current performance\n(ranking skill={_pop_at_full:.3f}, N={n_full})",
        xy=(n_full, _pop_at_full),
        xytext=(n_full + 500, _pop_at_full + 0.01),
        fontsize=BASE_FS * 1.08,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.5),
        color="grey",
    )
    # -- N=5000 projected star + label -------------------------------------
    _n_iati = 5000
    _pop_iati = float(a_pop + b_pop * np.log(_n_iati))
    ax_pop.scatter([_n_iati], [_pop_iati], color="tomato", zorder=6, s=120, marker="*")
    ax_pop.annotate(
        f"Estimated performance using\nall available IATI data\n(N={_n_iati}, ranking skill={_pop_iati:.3f})",
        xy=(_n_iati, _pop_iati),
        xytext=(_n_iati - 2400, _pop_iati - 0.02),
        fontsize=BASE_FS * 1.08,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5),
        color="tomato",
    )

    ax_pop.set_xlabel("Training set size (n)", fontsize=BASE_FS)
    ax_pop.set_ylabel("Ranking skill", fontsize=BASE_FS)
    ax_pop.set_title(
        f"Learning curve -- ranking skill ({set_label} set)", fontsize=BASE_FS
    )
    ax_pop.legend(fontsize=BASE_FS * 0.72, loc="lower right")
    ax_pop.tick_params(axis="both", labelsize=BASE_FS * 0.82)
    ax_pop.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = Path(__file__).resolve().parent / f"learning_curve_org_pop{_lc_suffix}.png"
    fig2.savefig(out2, dpi=120)
    plt.close(fig2)
    print(f"Plot 2 saved -> {out2}")

    if r2_only:
        return {
            "r2_at_n_full": float(a_r2 + b_r2 * np.log(n_full)),
            "inv_rmse_at_n_full": float(a_inv + b_inv * np.log(n_full)),
            "r2_proj_5x": float(r2_proj),
            "inv_rmse_proj_5x": float(inv_proj),
            "r2_gain": float(r2_proj - (a_r2 + b_r2 * np.log(n_full))),
            "inv_rmse_gain": float(inv_proj - (a_inv + b_inv * np.log(n_full))),
            "b_r2": float(b_r2),
            "b_inv": float(b_inv),
            "pop_at_n_full": pop_at_n_full,
            "pop_proj_5x": float(pop_proj),
            "pop_gain": float(pop_proj - pop_at_n_full),
            "b_pop": float(b_pop),
            "n_full": int(n_full),
            "n_proj": int(n_proj),
        }

    return {
        "r2_at_n_full": float(a_r2 + b_r2 * np.log(n_full)),
        "inv_rmse_at_n_full": float(a_inv + b_inv * np.log(n_full)),
        "r2_proj_5x": float(r2_proj),
        "inv_rmse_proj_5x": float(inv_proj),
        "r2_gain": float(r2_proj - (a_r2 + b_r2 * np.log(n_full))),
        "inv_rmse_gain": float(inv_proj - (a_inv + b_inv * np.log(n_full))),
        "b_r2": float(b_r2),
        "b_inv": float(b_inv),
        "pop_at_n_full": pop_at_n_full,
        "pop_proj_5x": float(pop_proj),
        "pop_gain": float(pop_proj - pop_at_n_full),
        "b_pop": float(b_pop),
        "n_full": int(n_full),
        "n_proj": int(n_proj),
    }


# ---------------------------------------------------------------------------
# Shared metric helper
# ---------------------------------------------------------------------------


def _metrics(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 2:
        return {
            k: np.nan
            for k in [
                "R2",
                "RMSE",
                "MAE",
                "SideAcc",
                "AccInt",
                "Spearman",
                "Pairwise",
                "N",
            ]
        }
    return {
        "R2": r2_metric(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAE": mae(yt, yp),
        "SideAcc": side_accuracy(yt, yp, threshold=SIDE_THRESHOLD),
        "AccInt": true_hit_accuracy(yt, yp),
        "Spearman": spearman_correlation(yt, yp),
        "Pairwise": pairwise_ordering_prob_excl_ties(yt, yp),
        "N": int(len(yt)),
    }


# ---------------------------------------------------------------------------
# Analysis 6: K-fold cross-validation on train+val (temporal folds)
# ---------------------------------------------------------------------------


def _load_embeddings_for_kfold(emb_path: Path) -> dict:
    """
    Load raw activity embeddings from JSONL. Returns dict: activity_id -> np.float32 array.
    Used by analysis_6 to refit PCA+UMAP per fold.
    """
    import json as _json

    if not emb_path.exists():
        print(
            f"    WARNING: embeddings file not found ({emb_path.name}) -- "
            "UMAP will not be refit per fold (pre-computed UMAP features used as-is)."
        )
        return {}

    emb_dict = {}
    with open(emb_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            aid = str(obj.get("activity_id", "")).strip()
            emb = obj.get("embedding")

            if aid and isinstance(emb, list) and len(emb) > 0:
                emb_dict[aid] = np.array(emb, dtype=np.float32)

    return emb_dict


def _refit_umap_for_fold(
    data: pd.DataFrame,
    tr_idx,
    emb_dict: dict,
    pca_dims: int = 50,
    n_neighbors: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Refit PCA + UMAP 3D on training fold only, then project ALL activities.

    This is the correct temporal isolation: the manifold is learned from
    training-fold embeddings only; test-fold activities are projected via
    transform() into that space, never influencing the fit.

    Returns a copy of data with umap3_x/y/z and umap_missing updated.
    """
    from sklearn.decomposition import PCA

    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError(
            "umap-learn is required for analysis_6. Install with: pip install umap-learn"
        )

    # Training fold embeddings (only those that have an embedding)
    tr_ids_with_emb = [str(aid) for aid in tr_idx if str(aid) in emb_dict]
    if len(tr_ids_with_emb) < 10:
        print(
            f"    WARNING: only {len(tr_ids_with_emb)} train activities have embeddings "
            f"-- UMAP refit unreliable, skipping update."
        )
        return data

    X_train_raw = np.stack([emb_dict[aid] for aid in tr_ids_with_emb])

    # All data activities that have embeddings
    all_ids_with_emb = [str(aid) for aid in data.index if str(aid) in emb_dict]
    X_all_raw = np.stack([emb_dict[aid] for aid in all_ids_with_emb])

    # Fit PCA on train, transform all
    n_components = min(pca_dims, X_train_raw.shape[1], X_train_raw.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(X_train_raw)
    X_train_pca = pca.transform(X_train_raw)
    X_all_pca = pca.transform(X_all_raw)

    # Fit UMAP on train PCA, transform all
    reducer = umap_lib.UMAP(
        n_components=3,
        n_neighbors=min(n_neighbors, len(tr_ids_with_emb) - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    reducer.fit(X_train_pca)
    U3_all = reducer.transform(X_all_pca)  # shape (n_all, 3)

    aid_to_umap = {aid: U3_all[i] for i, aid in enumerate(all_ids_with_emb)}

    data_fold = data.copy()
    nan3 = [np.nan, np.nan, np.nan]
    data_fold["umap3_x"] = [
        aid_to_umap.get(str(aid), nan3)[0] for aid in data_fold.index
    ]
    data_fold["umap3_y"] = [
        aid_to_umap.get(str(aid), nan3)[1] for aid in data_fold.index
    ]
    data_fold["umap3_z"] = [
        aid_to_umap.get(str(aid), nan3)[2] for aid in data_fold.index
    ]
    data_fold["umap_missing"] = (
        data_fold[["umap3_x", "umap3_y", "umap3_z"]].isna().any(axis=1).astype(float)
    )
    return data_fold


def analysis_6_temporal_kfold(
    data: pd.DataFrame,
    y: pd.Series,
    train_idx,
    val_idx,
    feature_cols: list,
    K: int = 6,
    fold_width_months: int = 18,
    min_train: int = 50,
    min_test: int = 20,
    label: str = "",
) -> None:
    """
    Temporal K-fold CV on train+val only (held_idx is never passed in or used).

    UMAP: refitted from scratch on each fold's training activities only, then
    test activities are projected via transform(). This prevents the manifold
    from being informed by future (test-fold) activity embeddings.

    LLM blend: omitted. LLM predictions only cover val-set activities -- there
    is insufficient coverage on training-fold activities to fit the residual
    corrector reliably.

    Methods compared (all RF+ExtraTrees, no LLM):
      per_org_mode          -- baseline (integer ratings)
      pred_rf               -- RF continuous prediction
      pred_rf_rounded       -- RF rounded to nearest integer (apples-to-apples vs baseline)
      pred_rf_corrected     -- RF + start-year linear drift correction (fit on train fold only)
      pred_rf_corr_rounded  -- RF + correction, rounded

    Fold construction:
      pool = train_idx union val_idx sorted by start_date.
      First boundary = earliest date with >= min_train activities before it.
      K boundaries spaced fold_width_months apart.
      Fold k: train = start_date < boundary_k
               test  = start_date in [boundary_k, boundary_k + fold_width_months)
      Folds with < min_train train or < min_test test activities are skipped.
    """
    print("\n" + "=" * 80)
    title = f"ANALYSIS 6: TEMPORAL K-FOLD CV (train+val only){' -- ' + label if label else ''}"
    print(title)
    print(
        f"  K={K} folds, fold_width={fold_width_months} months, "
        f"min_train={min_train}, min_test={min_test}"
    )
    print("  UMAP: refit on each train fold -- test points projected via transform()")
    print("  LLM blend: omitted (insufficient training-fold coverage)")
    print("=" * 80)

    # ---- Load raw embeddings once (needed for per-fold UMAP refit) ----
    emb_path = DATA_DIR / "outputs_targets_embeddings.jsonl"
    print(f"\nLoading raw embeddings from {emb_path.name}...")
    emb_dict = _load_embeddings_for_kfold(emb_path)
    print(f"  Loaded {len(emb_dict)} embeddings.")

    # ---- Pool and sort ----
    trainval_idx = train_idx.append(val_idx)
    pool = data.loc[trainval_idx].sort_values("start_date")
    pool_dates = pool["start_date"]
    min_date = pool_dates.min()
    max_date = pool_dates.max()
    print(
        f"  Train+val date range: {min_date.date()} -> {max_date.date()}  "
        f"(n={len(pool)})"
    )

    # ---- First boundary: earliest date with >= min_train activities before it ----
    sorted_dates = pool_dates.sort_values()
    first_boundary = None
    for i, d in enumerate(sorted_dates):
        if i >= min_train:
            first_boundary = d
            break
    if first_boundary is None:
        print(f"  Too few activities ({len(pool)}) to form any fold. Aborting.")
        return
    print(f"  First boundary: {first_boundary.date()}")

    # ---- K boundaries spaced fold_width_months apart ----
    boundaries = []
    for k in range(K):
        b = first_boundary + pd.DateOffset(months=k * fold_width_months)
        if b >= max_date:
            break
        boundaries.append(b)
    print(
        f"  Attempting {len(boundaries)} boundaries: "
        + ", ".join(b.strftime("%Y-%m") for b in boundaries)
    )

    # ---- Fold loop ----
    FOLD_METHODS = [
        ("per_org_mode", "Mode baseline (per-org)"),
        ("pred_rf", "RF continuous"),
        ("pred_rf_rounded", "RF rounded"),
        ("pred_rf_corrected", "RF + year correction"),
        ("pred_rf_corr_rounded", "RF + year corr, rounded"),
        ("pred_ridge", "Ridge GLM"),
        ("pred_ensemble", "Ensemble"),
    ]
    baseline_col = "per_org_mode"

    fold_rows = []
    org_fold_rows = []
    skipped = []

    # Orgs to break down separately (short_name, full_reporting_orgs_string, methods)
    # methods = list of (pred_col, method_label) to compare
    _bmz_full = next(o for o in KEEP_REPORTING_ORGS if "BMZ" in o)
    _org_methods = [("pred_rf_corrected", "RF+corr"), ("pred_ensemble", "Ensemble")]
    ORG_SUBSETS_A6 = [
        ("World Bank", "World Bank", _org_methods),
        ("BMZ", _bmz_full, _org_methods),
    ]

    # Feature cols for kfold: add org one-hots so models don't have to learn the
    # org-level mean from scratch -- they learn the delta from the org training mode.
    _org_cols = [
        f"rep_org_{i}"
        for i in range(NUM_ORGS_KEEP)
        if f"rep_org_{i}" not in feature_cols
    ]
    kfold_feature_cols = feature_cols + _org_cols

    for k, boundary in enumerate(boundaries):
        window_end = boundary + pd.DateOffset(months=fold_width_months)

        tr_idx_k = pool.loc[pool_dates < boundary].index
        te_idx_k = pool.loc[(pool_dates >= boundary) & (pool_dates < window_end)].index

        n_tr, n_te = len(tr_idx_k), len(te_idx_k)
        print(f"\n{'-'*70}")
        print(
            f"Fold {k+1}: train <{boundary.strftime('%Y-%m')}  "
            f"test [{boundary.strftime('%Y-%m')}, {window_end.strftime('%Y-%m')})  "
            f"n_train={n_tr}  n_test={n_te}"
        )

        if n_tr < min_train:
            print(f"  Skipping: n_train={n_tr} < {min_train}")
            skipped.append(k + 1)
            continue
        if n_te < min_test:
            print(f"  Skipping: n_test={n_te} < {min_test}")
            skipped.append(k + 1)
            continue

        # Refit UMAP on this fold's training activities only
        print(f"  Refitting PCA+UMAP on {n_tr} train activities...")
        d = _refit_umap_for_fold(data, tr_idx_k, emb_dict)

        # Per-org mode baseline (fit on training fold only; fills train + test rows)
        add_per_org_mode_baseline(
            d,
            y,
            tr_idx_k,
            te_idx_k,
            org_cols=[f"rep_org_{i}" for i in range(NUM_ORGS_KEEP)],
            out_col="per_org_mode",
        )

        # y_delta: subtract the training-set per-org mode so models learn the delta.
        # per_org_mode is already populated for both train and test rows.
        all_fold_idx = tr_idx_k.append(te_idx_k)
        y_delta = y.copy().astype(float)
        y_delta.loc[all_fold_idx] -= d.loc[all_fold_idx, "per_org_mode"]
        d["rating_delta"] = y_delta

        # RF + ExtraTrees (trained on delta, prediction converted back to absolute)
        print(f"  Training RF+ExtraTrees on y_delta...")
        rf_delta, _ = run_random_forest_median_impute_noclip(
            data=d,
            feature_cols=kfold_feature_cols,
            target_col="rating_delta",
            train_index=tr_idx_k,
            rf_params=RF_PARAMS,
            ensemble_with_extratrees=True,
        )
        d["pred_rf"] = np.clip(
            pd.Series(rf_delta, index=d.index) + d["per_org_mode"], 0.0, 5.0
        )
        d["pred_rf_rounded"] = d["pred_rf"].round()

        # RF + start-year correction (in absolute space -- y and pred are both absolute)
        d["pred_rf_corrected"] = d["pred_rf"].copy()
        apply_start_year_trend_correction(d, y, tr_idx_k, pred_col="pred_rf_corrected")
        d["pred_rf_corr_rounded"] = d["pred_rf_corrected"].round()

        # Ridge GLM (trained on delta, converted back to absolute)
        print(f"  Training Ridge GLM on y_delta...")
        ridge_delta, _ = run_ridge_glm_median_impute_noclip(
            data=d,
            feature_cols=kfold_feature_cols,
            target_col="rating_delta",
            train_index=tr_idx_k,
        )
        d["pred_ridge"] = np.clip(
            pd.Series(ridge_delta, index=d.index) + d["per_org_mode"], 0.0, 5.0
        )
        d["pred_ensemble"] = 0.5 * (d["pred_rf"] + d["pred_ridge"])
        apply_start_year_trend_correction(d, y, tr_idx_k, pred_col="pred_ensemble")

        # Ridge baseline: risks + org dummies only -- also trained on delta
        baseline_feat_cols = ["risks"] + [
            f"rep_org_{i}" for i in range(NUM_ORGS_KEEP - 1)
        ]
        risks_ridge_delta, _ = run_ridge_glm_median_impute_noclip(
            data=d,
            feature_cols=baseline_feat_cols,
            target_col="rating_delta",
            train_index=tr_idx_k,
        )
        d["ridge_baseline_risk_org"] = np.clip(
            pd.Series(risks_ridge_delta, index=d.index) + d["per_org_mode"], 0.0, 5.0
        )

        # Evaluate on test fold
        yt = y.loc[te_idx_k].astype(float).to_numpy()
        base_m = _metrics(yt, d.loc[te_idx_k, baseline_col].astype(float).to_numpy())
        base_r2 = base_m["R2"]
        base_pw = base_m["Pairwise"]

        # Groups for within-group pairwise ordering (org + start_year)
        all_groups_k = (
            d.loc[te_idx_k, "reporting_orgs"].astype(str)
            + "_"
            + d.loc[te_idx_k, "start_date"].dt.year.astype(str)
        ).to_numpy()

        print(f"  {'Method':<35} {'R2':>8} {'DeltaR2':>9} {'PW':>8} {'DeltaPW':>9}")
        print(f"  {'-'*72}")

        for col, label in FOLD_METHODS:
            m = _metrics(yt, d.loc[te_idx_k, col].astype(float).to_numpy())
            r2_adv = m["R2"] - base_r2
            pw_adv = m["Pairwise"] - base_pw
            flag = "  <- below baseline!" if col != baseline_col and r2_adv < 0 else ""
            print(
                f"  {label:<35} {m['R2']:>8.4f} {r2_adv:>+9.4f} "
                f"{m['Pairwise']:>8.4f} {pw_adv:>+9.4f}{flag}"
            )
            yp_col = d.loc[te_idx_k, col].astype(float).to_numpy()
            wg_pop = within_group_pairwise_ordering_prob(yt, yp_col, all_groups_k)[
                "prob"
            ]
            fold_rows.append(
                {
                    "fold": k + 1,
                    "col": col,
                    "label": label,
                    "R2": m["R2"],
                    "R2_adv": r2_adv,
                    "PW": m["Pairwise"],
                    "PW_adv": pw_adv,
                    "RMSE": m["RMSE"],
                    "WG_POP": wg_pop,
                    "N": m["N"],
                    "n_train": n_tr,
                }
            )

        # ---- Per-org breakdown: within-BMZ and within-WB ----
        # POP uses _stratified_pairwise_prob with start_year as group key,
        # so only pairs that started in the same year are counted.
        if "start_year" not in d.columns:
            d = d.copy()
            d["start_year"] = d["start_date"].dt.year

        print(
            f"\n  Per-org breakdown (R2 + same-year-pair POP, both methods vs baseline):"
        )
        print(
            f"  {'Org':<12} {'Method':<18} {'R2':>8} {'DeltaR2':>9} {'POP(yr)':>9} {'DeltaPOP(yr)':>10} {'n':>4}"
        )
        print(f"  {'-'*95}")

        for org_short, org_full, methods in ORG_SUBSETS_A6:
            org_mask = d.loc[te_idx_k, "reporting_orgs"] == org_full
            org_te_idx = te_idx_k[org_mask.values]
            n_org = len(org_te_idx)

            if n_org < 2:
                print(f"  {org_short:<12}  n={n_org}, skipping")
                continue

            yt_org = y.loc[org_te_idx].astype(float).to_numpy()
            base_org = (
                d.loc[org_te_idx, "ridge_baseline_risk_org"].astype(float).to_numpy()
            )
            mode_org = d.loc[org_te_idx, "per_org_mode"].astype(float).to_numpy()
            sy_org = d.loc[org_te_idx, "start_year"].astype(str).to_numpy()

            base_metrics_org = _metrics(yt_org, base_org)
            base_r2_org = base_metrics_org["R2"]
            base_rmse_org = base_metrics_org["RMSE"]
            base_mae_org = base_metrics_org["MAE"]
            base_pop_yr = within_group_pairwise_ordering_prob(yt_org, base_org, sy_org)[
                "prob"
            ]
            mode_metrics_org = _metrics(yt_org, mode_org)
            mode_rmse_org = mode_metrics_org["RMSE"]
            mode_mae_org = mode_metrics_org["MAE"]

            print(
                f"  {org_short:<12} {'risks baseline':<18} {base_r2_org:>8.4f} {'':>9} "
                f"{base_pop_yr:>9.4f} {'':>10} {n_org:>4}"
            )

            for pred_col, method_label in methods:
                if pred_col not in d.columns:
                    continue
                rf_org = d.loc[org_te_idx, pred_col].astype(float).to_numpy()
                rf_metrics_org = _metrics(yt_org, rf_org)
                rf_r2_org = rf_metrics_org["R2"]
                rf_rmse_org = rf_metrics_org["RMSE"]
                rf_mae_org = rf_metrics_org["MAE"]
                _r_rf = within_group_pairwise_ordering_prob(yt_org, rf_org, sy_org)
                rf_pop_yr, n_pairs_yr = _r_rf["prob"], _r_rf["n_pairs"]

                r2_adv_org = rf_r2_org - base_r2_org
                pop_adv_org = (
                    (rf_pop_yr - base_pop_yr)
                    if (np.isfinite(rf_pop_yr) and np.isfinite(base_pop_yr))
                    else np.nan
                )

                print(
                    f"  {org_short:<12} {method_label:<18} {rf_r2_org:>8.4f} {r2_adv_org:>+9.4f} "
                    f"{rf_pop_yr:>9.4f} {pop_adv_org:>+10.4f} {n_org:>4}  ({n_pairs_yr} yr-pairs)"
                )

                org_fold_rows.append(
                    {
                        "fold": k + 1,
                        "org": org_short,
                        "method": method_label,
                        "R2_base": base_r2_org,
                        "R2_rf": rf_r2_org,
                        "R2_adv": r2_adv_org,
                        "RMSE_mode": mode_rmse_org,
                        "MAE_mode": mode_mae_org,
                        "RMSE_base": base_rmse_org,
                        "RMSE_rf": rf_rmse_org,
                        "MAE_base": base_mae_org,
                        "MAE_rf": rf_mae_org,
                        "POP_yr_base": base_pop_yr,
                        "POP_yr_rf": rf_pop_yr,
                        "POP_yr_adv": pop_adv_org,
                        "n": n_org,
                        "n_pairs_yr": n_pairs_yr,
                        "n_train": n_tr,
                    }
                )

    if not fold_rows:
        print("\nNo valid folds completed.")
        return [], []

    # ---- Summary ----
    df_folds = pd.DataFrame(fold_rows)
    n_folds_run = df_folds["fold"].nunique()

    print(f"\n{'='*80}")
    print(
        f"SUMMARY: mean +/- SD of (method - baseline) across {n_folds_run} valid folds"
    )
    if skipped:
        print(f"  Skipped folds: {skipped}")
    print(f"{'='*80}")

    hdr = (
        f"  {'Method':<35} {'mean DeltaR^2':>10} {'SD DeltaR^2':>8} "
        f"{'mean DeltaPW':>10} {'SD DeltaPW':>8} {'folds':>6} {'min DeltaR^2':>9}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for col, label in FOLD_METHODS[1:]:  # skip baseline itself
        sub = df_folds[df_folds["col"] == col]
        if sub.empty:
            continue
        r2_advs = sub["R2_adv"].to_numpy(dtype=float)
        pw_advs = sub["PW_adv"].to_numpy(dtype=float)
        mean_r2 = float(np.nanmean(r2_advs))
        sd_r2 = (
            float(np.nanstd(r2_advs, ddof=1))
            if np.sum(np.isfinite(r2_advs)) > 1
            else np.nan
        )
        mean_pw = float(np.nanmean(pw_advs))
        sd_pw = (
            float(np.nanstd(pw_advs, ddof=1))
            if np.sum(np.isfinite(pw_advs)) > 1
            else np.nan
        )
        min_r2 = float(np.nanmin(r2_advs))
        stable = (
            " ok" if mean_r2 > 0 and (np.isnan(sd_r2) or sd_r2 < abs(mean_r2)) else " ?"
        )
        print(
            f"  {label:<35} {mean_r2:>+10.4f} {sd_r2:>8.4f} "
            f"{mean_pw:>+10.4f} {sd_pw:>8.4f} {len(r2_advs):>6} {min_r2:>+9.4f}{stable}"
        )

    # Worst fold for corrected+rounded (the cleanest apples-to-apples with baseline)
    best_sub = df_folds[df_folds["col"] == "pred_rf_corr_rounded"].sort_values("R2_adv")
    if not best_sub.empty:
        worst = best_sub.iloc[0]
        neg = best_sub[best_sub["R2_adv"] < 0]
        print(
            f"\n  Worst fold (RF+corr+rounded): fold {int(worst['fold'])}  "
            f"DeltaR^2={worst['R2_adv']:+.4f}  R^2={worst['R2']:.4f}  n_test={int(worst['N'])}"
        )
        if not neg.empty:
            print(
                f"  *** WORSE than baseline in fold(s): "
                + ", ".join(str(int(f)) for f in neg["fold"])
                + " ***"
            )
        else:
            print(f"  Beats baseline in all {n_folds_run} valid folds.")

    print(f"\n  ok = mean > 0 and SD < |mean| (advantage stable across folds).")
    print(
        f"  Rounded variants are the apples-to-apples comparison with per-org mode (also integer)."
    )

    # ---- Per-org summary across folds ----
    if org_fold_rows:
        df_org = pd.DataFrame(org_fold_rows)
        print(f"\n{'='*80}")
        print(
            "ANALYSIS 6 -- PER-ORG SUMMARY (both methods vs baseline, same-year-pair POP)"
        )
        print(f"  Only pairs started in the same calendar year count towards POP.")
        print(f"{'='*80}")
        hdr_org = (
            f"  {'Org':<12} {'Method':<18} {'mean DeltaR^2':>10} {'SD DeltaR^2':>8} "
            f"{'mean DeltaPOP(yr)':>14} {'SD DeltaPOP(yr)':>12} {'folds':>6} {'mean n/fold':>11}"
        )
        print(hdr_org)
        print("  " + "-" * (len(hdr_org) - 2))
        for org_short in df_org["org"].unique():
            for method_label in df_org["method"].unique():
                sub = df_org[
                    (df_org["org"] == org_short) & (df_org["method"] == method_label)
                ]
                if sub.empty:
                    continue
                r2_advs = sub["R2_adv"].dropna().to_numpy(dtype=float)
                pop_advs = sub["POP_yr_adv"].dropna().to_numpy(dtype=float)
                mean_r2 = float(np.mean(r2_advs)) if len(r2_advs) else np.nan
                sd_r2 = float(np.std(r2_advs, ddof=1)) if len(r2_advs) > 1 else np.nan
                mean_pop = float(np.mean(pop_advs)) if len(pop_advs) else np.nan
                sd_pop = (
                    float(np.std(pop_advs, ddof=1)) if len(pop_advs) > 1 else np.nan
                )
                mean_n = float(sub["n"].mean())
                stable = (
                    " ok"
                    if mean_r2 > 0 and (np.isnan(sd_r2) or sd_r2 < abs(mean_r2))
                    else " ?"
                )
                print(
                    f"  {org_short:<12} {method_label:<18} {mean_r2:>+10.4f} {sd_r2:>8.4f} "
                    f"{mean_pop:>+14.4f} {sd_pop:>12.4f} {len(r2_advs):>6} {mean_n:>11.1f}{stable}"
                )

        # Also print raw fold values for transparency
        print(f"\n  Fold-by-fold detail:")
        print(
            f"  {'fold':>5} {'org':<12} {'method':<18} {'R2_base':>9} {'R2_rf':>8} {'DeltaR2':>8}"
            f" {'POP_base':>9} {'POP_rf':>8} {'DeltaPOP':>8} {'n':>4} {'yr-pairs':>9}"
        )
        print(f"  {'-'*115}")
        for row in org_fold_rows:
            pop_adv_str = (
                f"{row['POP_yr_adv']:>+8.4f}"
                if np.isfinite(row.get("POP_yr_adv", np.nan))
                else "     nan"
            )
            pop_base_str = (
                f"{row['POP_yr_base']:>9.4f}"
                if np.isfinite(row.get("POP_yr_base", np.nan))
                else "      nan"
            )
            pop_rf_str = (
                f"{row['POP_yr_rf']:>8.4f}"
                if np.isfinite(row.get("POP_yr_rf", np.nan))
                else "     nan"
            )
            print(
                f"  {row['fold']:>5} {row['org']:<12} {row['method']:<18} "
                f"{row['R2_base']:>9.4f} {row['R2_rf']:>8.4f} {row['R2_adv']:>+8.4f} "
                f"{pop_base_str} {pop_rf_str} {pop_adv_str} "
                f"{row['n']:>4} {row['n_pairs_yr']:>9}"
            )

    return fold_rows, org_fold_rows


# ---------------------------------------------------------------------------
# Analysis 7: Leave-one-org-out -- full data build + training helpers
# ---------------------------------------------------------------------------


def build_model_data() -> tuple:
    """
    Build the full feature matrix and splits. Mirrors A_overall_rating_fit_and_evaluate.main().
    Returns (data, y, train_idx, val_idx, held_idx, feature_cols, llm_pred_cols).
    WARNING: slow (~3-5 min on first call). The held-out index is returned so
    callers can confirm they never use it, but nothing in this file evaluates on it.
    """
    print(
        "\nBuilding full model data (mirrors A_overall_rating_fit_and_evaluate pipeline)..."
    )
    s = str(INFO_CSV)

    grades_df = load_grades(str(DATA_DIR / "*_grades.jsonl"))
    scope_df = load_activity_scope(s)
    impl_type_df = load_implementing_org_type(s)
    gdp_df = load_gdp_percap(s)
    expend_df = load_llm_planned_expenditure()
    world_bank_df = load_world_bank_indicators(s)
    region_cols = [
        "region_AFE",
        "region_AFW",
        "region_EAP",
        "region_ECA",
        "region_LAC",
        "region_MENA",
        "region_SAS",
    ]
    regions_df = pd.read_csv(
        s, usecols=["activity_id"] + region_cols, index_col="activity_id"
    )
    is_completed = load_is_completed(s)
    lod_df = load_loan_or_disbursement()
    ratings = load_ratings(str(DATA_DIR / "merged_overall_ratings.jsonl"))
    tc_maps_df = load_targets_context_maps_features(
        DATA_DIR / "outputs_targets_context_maps.jsonl"
    )
    sector_clusters_df = data_sector_clusters(
        str(
            DATA_DIR
            / "outputs_finance_sectors_disbursements_baseline_gemini2p5flash.jsonl"
        )
    )
    data = ratings.to_frame(name="rating")
    for df in [
        is_completed,
        grades_df,
        scope_df,
        gdp_df,
        expend_df,
        world_bank_df,
        regions_df,
        lod_df,
        impl_type_df,
        tc_maps_df,
        sector_clusters_df,
    ]:
        data = data.join(df, how="left")

    data["activity_scope"] = pd.to_numeric(data["activity_scope"], errors="coerce")
    for col in region_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0.0)

    data, _ = add_similarity_features(data, s, KEEP_REPORTING_ORGS)
    data = restrict_to_reporting_orgs_exact(data, KEEP_REPORTING_ORGS)
    data = add_dates_to_dataframe(data, s)
    data["planned_duration"] = load_llm_planned_duration()["planned_duration"].reindex(
        data.index
    )
    data["has_start"] = data["start_date"].notna()

    llm_unc_feats = [
        "finance",
        "integratedness",
        "implementer_performance",
        "targets",
        "context",
        "risks",
        "complexity",
    ]
    data = add_enhanced_uncertainty_features(data, llm_unc_feats)

    feature_cols = [
        "finance",
        "integratedness",
        "implementer_performance",
        "targets",
        "context",
        "risks",
        "complexity",
        "activity_scope",
        "gdp_percap",
        "cpia_score",
        *WGI_COLS,
        "region_AFE",
        "region_AFW",
        "region_EAP",
        "region_ECA",
        "region_LAC",
        "region_MENA",
        "region_SAS",
        "finance_is_loan",
        "planned_duration",
        "planned_expenditure",
        *[f"rep_org_{i}" for i in range(NUM_ORGS_KEEP - 1)],
        "umap3_x",
        "umap3_y",
        "umap3_z",
        "sector_distance",
        "country_distance",
        *[c for c in data.columns if c.startswith("sector_cluster_")],
        "llm_features_missing_count",
        "llm_features_present_ratio",
        "governance_missing_count",
        "feature_completeness_ratio",
        "cpia_missing",
        "sector_clusters_missing",
        "gdp_percap_missing",
        "planned_expenditure_missing",
        "planned_duration_missing",
        "wgi_any_missing",
        "umap_missing",
    ]
    feature_cols = [f for f in feature_cols if f in data.columns]

    # Interaction features
    if "planned_expenditure" in data.columns and "complexity" in data.columns:
        data["expenditure_x_complexity"] = (
            data["planned_expenditure"] * data["complexity"]
        )
    if "planned_expenditure" in data.columns and "planned_duration" in data.columns:
        data["expenditure_per_year_log"] = np.log(
            (data["planned_expenditure"] / data["planned_duration"]).where(
                (data["planned_duration"] >= 1)
                & (data["planned_expenditure"] >= 100000),
                np.nan,
            )
        )

    complete_mask = data["is_completed"].fillna(0).astype(int) == 1
    data = data.loc[complete_mask]

    if data["reporting_orgs"].isna().any():
        raise ValueError(
            f"reporting_orgs has {data['reporting_orgs'].isna().sum()} NaNs"
        )
    if data["start_date"].isna().any():
        raise ValueError(f"start_date has {data['start_date'].isna().sum()} NaNs")

    _d = data.sort_values("start_date")
    _d = _d[_d["start_date"] < pd.to_datetime(TOO_LATE_CUTOFF)]
    train_idx = _d[_d["start_date"] <= pd.to_datetime(LATEST_TRAIN_POINT)].index
    val_idx = _d[
        (_d["start_date"] > pd.to_datetime(LATEST_TRAIN_POINT))
        & (_d["start_date"] <= pd.to_datetime(LATEST_VALIDATION_POINT))
    ].index
    held_idx = _d[_d["start_date"] > pd.to_datetime(LATEST_VALIDATION_POINT)].index

    from split_constants import assert_split_matches_canonical

    assert_split_matches_canonical(train_idx, val_idx, held_idx, splits_csv=SPLITS_CSV)

    y = data["rating"].astype(float)

    min_date = data["start_date"].min()
    data["start_date_days"] = (data["start_date"] - min_date).dt.days

    # Feature engineering (same as A_overall_rating_fit_and_evaluate, applied to full data)
    if all(c in data.columns for c in WGI_COLS):
        data["governance_composite"] = data[WGI_COLS].mean(axis=1)
    if "governance_composite" in data.columns and "complexity" in data.columns:
        data["governance_x_complexity"] = (
            data["governance_composite"] * data["complexity"]
        )
    if "planned_expenditure" in data.columns:
        data["log_planned_expenditure"] = data["planned_expenditure"]
        data["planned_expenditure"] = np.exp(data["planned_expenditure"])
    if "planned_expenditure" in data.columns and "planned_duration" in data.columns:
        data["expenditure_per_year"] = (
            data["planned_expenditure"] / data["planned_duration"]
        ).where(
            (data["planned_duration"] >= 1) & (data["planned_expenditure"] >= 100_000),
            np.nan,
        )
        data["expenditure_per_year_log"] = np.log(
            data["expenditure_per_year"].where(data["expenditure_per_year"] > 0)
        )

    # governance_composite replaces individual WGI cols
    if "governance_composite" in data.columns:
        feature_cols = [f for f in feature_cols if f not in WGI_COLS]
    new_features = [
        f
        for f in [
            "governance_composite",
            "governance_x_complexity",
            "expenditure_x_complexity",
            "expenditure_per_year_log",
            "log_planned_expenditure",
        ]
        if f in data.columns
    ]
    feature_cols = feature_cols + [f for f in new_features if f not in feature_cols]

    # LLM predictions
    PREDICTION_CONFIGS = get_llm_prediction_configs()
    llm_pred_cols = []
    for name, path, parser, _label in PREDICTION_CONFIGS:
        series_name = f"{name}_prediction"
        if not Path(path).exists():
            print(f"  Skipping {Path(path).name} (file not found)")
            continue
        preds = (
            load_predictions_from_jsonl(path, parser, series_name)
            .groupby(level=0)
            .mean()
        )
        data = data.join(preds, how="left")
        if preds.notna().sum() > 0:
            llm_pred_cols.append(series_name)
    if llm_pred_cols:
        data["mean_all_llm_preds"] = data[llm_pred_cols].mean(axis=1)

    print(
        f"Full data ready: {len(data)} activities | "
        f"train {len(train_idx)} | val {len(val_idx)} | test {len(held_idx)}"
    )
    return data, y, train_idx, val_idx, held_idx, feature_cols, llm_pred_cols


def _train_and_predict_fold(
    data: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
    train_idx,
    eval_idx,
) -> pd.DataFrame:
    """
    Train all 4 methods on train_idx, write prediction columns into a copy of data.
    Returns the modified copy. eval_idx is passed to add_per_org_mode_baseline.
    """
    d = data.copy()

    # Feature cols augmented with org one-hots (same logic as analysis_6_temporal_kfold)
    _org_cols = [
        f"rep_org_{i}"
        for i in range(NUM_ORGS_KEEP)
        if f"rep_org_{i}" not in feature_cols
    ]
    feat_cols_with_orgs = feature_cols + _org_cols

    # Baseline 1: per-org mode (fit on train only; fills both train and eval rows)
    add_per_org_mode_baseline(
        d,
        y,
        train_idx,
        eval_idx,
        org_cols=[f"rep_org_{i}" for i in range(NUM_ORGS_KEEP)],
        out_col="per_org_mode",
    )

    # y_delta: subtract the training-set per-org mode so models learn the delta
    all_fold_idx = train_idx.append(eval_idx)
    y_delta = y.copy().astype(float)
    y_delta.loc[all_fold_idx] -= d.loc[all_fold_idx, "per_org_mode"]
    d["rating_delta"] = y_delta

    # Baseline 2: Ridge on risks + org dummies (trained on delta, back to absolute)
    baseline_feat_cols = ["risks"] + [f"rep_org_{i}" for i in range(NUM_ORGS_KEEP - 1)]
    ridge_delta, _ = run_ridge_glm_median_impute_noclip(
        data=d,
        feature_cols=baseline_feat_cols,
        target_col="rating_delta",
        train_index=train_idx,
    )
    d["ridge_baseline_risk_org"] = np.clip(
        pd.Series(ridge_delta, index=d.index) + d["per_org_mode"], 0.0, 5.0
    )

    # RF+ET (trained on delta, back to absolute)
    rf_delta, _ = run_random_forest_median_impute_noclip(
        data=d,
        feature_cols=feat_cols_with_orgs,
        target_col="rating_delta",
        train_index=train_idx,
        rf_params=RF_PARAMS,
        ensemble_with_extratrees=True,
    )
    d["pred_rf"] = np.clip(
        pd.Series(rf_delta, index=d.index) + d["per_org_mode"], 0.0, 5.0
    )

    # Full-feature Ridge (trained on delta, back to absolute; for ensemble)
    ridge_full_delta, _ = run_ridge_glm_median_impute_noclip(
        data=d,
        feature_cols=feat_cols_with_orgs,
        target_col="rating_delta",
        train_index=train_idx,
    )
    d["pred_ridge_full"] = np.clip(
        pd.Series(ridge_full_delta, index=d.index) + d["per_org_mode"], 0.0, 5.0
    )
    d["ridge_plus_rf"] = d[["pred_ridge_full", "pred_rf"]].mean(axis=1)

    # RF+ET + LLM + recency corrector
    add_rf_llm_residual_corrector(
        d,
        y,
        meta_train_idx=train_idx,
        rf_col="pred_rf",
        llm_col="mean_all_llm_preds",
        out_col="pred_rf_llm_modded",
        alpha=5.0,
        lam=1.0,
    )

    # Ridge+RF ensemble + LLM + recency corrector
    add_rf_llm_residual_corrector(
        d,
        y,
        meta_train_idx=train_idx,
        rf_col="ridge_plus_rf",
        llm_col="mean_all_llm_preds",
        out_col="ridge_plus_rf_llm_modded",
        alpha=5.0,
        lam=1.0,
    )

    # Start-year trend correction on both LLM-corrected columns
    apply_start_year_trend_correction(d, y, train_idx, pred_col="pred_rf_llm_modded")
    apply_start_year_trend_correction(
        d, y, train_idx, pred_col="ridge_plus_rf_llm_modded"
    )

    return d


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    print(
        "No flag given. Use --lc-pop (learning curve), --test (test set eval), "
        "or --lc-kfold (k-fold learning curve).",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    import sys

    _args = [a.lower() for a in sys.argv[1:]]
    _kfold_only = any(a in ("kfold", "--kfold-only") for a in _args)
    _test_only = any(a in ("test", "--test") for a in _args)
    _lc_pop = any(a in ("lc-pop", "--lc-pop") for a in _args)
    _lc_kfold = any(a in ("lc-kfold", "--lc-kfold") for a in _args)
    if _lc_kfold:
        full_data, y, train_idx, val_idx, held_idx, feature_cols, _ = build_model_data()
        fold_rows_18, org_rows_18 = analysis_6_temporal_kfold(
            full_data, y, train_idx, val_idx, feature_cols, label="18-month folds"
        )
        fold_rows_54, org_rows_54 = analysis_6_temporal_kfold(
            full_data,
            y,
            train_idx,
            val_idx,
            feature_cols,
            K=4,
            fold_width_months=54,
            min_test=100,
            label="~200-per-fold (54-month folds)",
        )
    elif _lc_pop:
        full_data, y, train_idx, val_idx, held_idx, feature_cols, _ = build_model_data()
        trainval_idx = train_idx.append(val_idx)
        analysis_17_learning_curve(
            full_data,
            y,
            train_idx,
            val_idx,
            feature_cols,
            train_pool_idx=trainval_idx,
            fixed_eval_idx=held_idx,
            set_label="test",
            r2_only=False,
        )
    elif _test_only:
        pass
    else:
        main()
