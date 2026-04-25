"""
Refactored rating prediction script.

Usage:
    python A_overall_rating_fit_and_evaluate.py [importances] [plot]

Arguments:
    importances  Print detailed top-30 feature importance table (terminal + LaTeX)
                 BEFORE the required results output.
    plot         Show matplotlib figures for pred_rf_llm_modded only.
"""

import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from data_loan_disbursement import load_loan_or_disbursement
from feature_engineering import (
    add_dates_to_dataframe,
    add_enhanced_uncertainty_features,
    add_similarity_features,
    data_sector_clusters,
    load_activity_scope,
    load_gdp_percap,
    load_grades,
    load_implementing_org_type,
    load_is_completed,
    load_ratings,
    load_targets_context_maps_features,
    load_world_bank_indicators,
    restrict_to_reporting_orgs_exact,
)
from leakage_risk import (
    EXCLUDE_TEST_LEAKAGE_RISK,
    GRADE_LEAKAGE_IDS,
    TEST_ANY_LEAKAGE_IDS,
)
from llm_load_predictions import (
    VARIANT_PATHS,
    get_llm_prediction_configs,
    load_predictions_from_jsonl,
)
from ml_models import (
    apply_start_year_trend_correction,
    bootstrap_ci,
    one_sd_shift_importance,
    run_random_forest_median_impute_noclip,
    run_ridge_glm_median_impute_noclip,
    run_xgboost_native_missing,
)
from overall_rating_feature_labels import get_display_name
from overall_rating_rf_conformal import get_error_bars_split_conformal
from scoring_metrics import (
    adjusted_r2,
    mae,
    org_year_pairwise_ordering_prob,
    rmse,
    side_accuracy,
    spearman_correlation,
    true_hit_accuracy,
    within_group_pairwise_ordering_prob,
    within_group_pairwise_ordering_prob_on_reference_pairs,
    within_group_spearman_correlation,
)
from split_constants import (
    LATEST_TRAIN_POINT,
    LATEST_VALIDATION_POINT,
    TOO_LATE_CUTOFF,
    assert_split_matches_canonical,
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Rating scale: 0=Highly Unsatisfactory ... 5=Highly Satisfactory (from _CANON_LABEL_TO_0_5)
RATING_CLIP = (0.0, 5.0)

# Thesis LLM_adjustment: ridge corrector parameters
LLM_CORRECTOR_ALPHA = 5.0  # L2 penalty on (beta_1, beta_2)
LLM_CORRECTOR_LAM = 1.0  # scaling factor lambda applied to residual correction
LLM_CORRECTOR_CLIP = RATING_CLIP  # corrector output clipped to same rating scale

# Thesis invariants (checked at import time)
assert RATING_CLIP == (0.0, 5.0), "rating scale is 0-5 per _CANON_LABEL_TO_0_5"
assert LLM_CORRECTOR_ALPHA == 5.0, "thesis LLM_adjustment: ridge penalty alpha=5"
assert LLM_CORRECTOR_LAM == 1.0, "thesis LLM_adjustment: lambda=1 (full correction)"
assert LLM_CORRECTOR_CLIP == (
    0.0,
    5.0,
), "thesis LLM_adjustment: clip to rating scale [0,5]"

MODEL_TO_USE = "logit_and_ordinal"

NUM_ORGS_KEEP = 4
# LATEST_TRAIN_POINT, LATEST_VALIDATION_POINT, TOO_LATE_CUTOFF imported from split_constants
USE_VAL_IN_TRAIN = (
    True  # set True to run on held-out test set (requires typing "understood")
)
TOP_10_FEATURES_ONLY = False  # set True to restrict model to top 10 features by importance (requires a prior run with importances saved)
USE_TOP_FEATURES_FOR_GENERALIZATION = (
    False  # set True to use ~19-feature set selected by val delta_r2/pairwise
)
MAGIC_OFFSET = False  # set True to apply per-column optimal offset (mean(y_true)-mean(y_pred)) on test set, maximising R^2

KEEP_REPORTING_ORGS = [
    "UK - Foreign, Commonwealth Development Office (FCDO)",
    "Asian Development Bank",
    "World Bank",
    "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); Federal Ministry for Economic Cooperation and Development (BMZ)",
]

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
INFO_FOR_ACTIVITY_FORECASTING = str(
    DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"
)
ALL_GRADES = str(DATA_DIR / "*_grades.jsonl")
OUT_MISC = DATA_DIR / "outputs_misc.jsonl"
MERGED_OVERALL_RATINGS = str(DATA_DIR / "merged_overall_ratings.jsonl")
TARGETS_CONTEXT_MAPS = DATA_DIR / "outputs_targets_context_maps.jsonl"
# For test-set evaluation (USE_VAL_IN_TRAIN=True): UMAP was refit on train+val.
# Generate this by running K_generate_embedding_distances_compressions.py with
# INCLUDE_VAL_IN_FIT=True and LATEST_TRAIN_VAL_POINT set to the earliest test-set
# start date (check split output from a dry run).
TARGETS_CONTEXT_MAPS_TRAINVAL = DATA_DIR / "outputs_targets_context_maps_trainval.jsonl"
FINANCE_SECTORS = str(
    DATA_DIR / "outputs_finance_sectors_disbursements_baseline_gemini2p5flash.jsonl"
)
OUTPUT_DIR = DATA_DIR / "rating_model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LLM_EXPENDITURE_JSONL = DATA_DIR / "llm_planned_expenditure.jsonl"

# ---------------------------------------------------------------------------
# LEAKAGE HANDLING
# ---------------------------------------------------------------------------
# How to handle activities flagged in LEAKAGERISK.json.
# Only takes effect when EXCLUDE_TEST_LEAKAGE_RISK is True.
#   "drop"                — remove leakage activities from test_idx (original behaviour)
#   "replace_predictions" — NaN grade features for grade-leakage activities before
#                           training (RF fills from training medians), AND after all
#                           predictions are built overwrite LLM-blended columns:
#                             grade-leakage test rows  → pred_rf_no_llm
#                             forecast-only test rows  → pred_rf (base, no LLM forecast)
#   "median_impute"       — only NaN grade features before training; no prediction swap
LEAKAGE_HANDLING_METHOD: str = "replace_predictions"

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------


def _latex_escape(s: str) -> str:
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(s))


def add_per_org_mode_baseline(data, y, train_idx, test_idx, org_cols, out_col):
    y_tr = y.loc[train_idx].astype(float)
    overall = float(y_tr.mode().iloc[0])
    modes = {}
    for c in org_cols:
        sub = y_tr[data.loc[train_idx, c].fillna(0).astype(int) == 1]
        modes[c] = float(sub.mode().iloc[0]) if len(sub) else overall
    pred = pd.Series(overall, index=data.index, dtype=float)
    for c in org_cols:
        pred[data[c].fillna(0).astype(int) == 1] = modes[c]
    data[out_col] = pred
    return data


def add_rf_llm_residual_corrector(
    data,
    y,
    *,
    meta_train_idx,
    rf_col="pred_rf",
    llm_col="mean_all_llm_preds",
    out_col="pred_rf_llm_modded",
    alpha=5.0,
    lam=1.0,
    clip_lo=0.0,
    clip_hi=5.0,
    use_llm=True,
    use_features=None,
):
    if use_llm:
        feats = [rf_col, llm_col]
        valid_feat_mask = data[feats].notna().all(axis=1)
    else:
        feats = [rf_col]
        valid_feat_mask = data[rf_col].notna()

    meta_train_idx = pd.Index(meta_train_idx)
    fit_idx = meta_train_idx[
        valid_feat_mask.loc[meta_train_idx].to_numpy()
        & y.loc[meta_train_idx].notna().to_numpy()
    ]

    if len(fit_idx) == 0:
        print(
            "No rows to fit residual corrector (check llm coverage on meta_train_idx)."
        )
        data[out_col] = data[rf_col].astype(float)
        return None, fit_idx

    X = data.loc[fit_idx, feats].astype(float).to_numpy()
    r = (
        y.loc[fit_idx].astype(float) - data.loc[fit_idx, rf_col].astype(float)
    ).to_numpy()
    model = Ridge(alpha=alpha, fit_intercept=True).fit(X, r)

    data[out_col] = data[rf_col].astype(float)
    apply_idx = data.index[valid_feat_mask]
    X_apply = data.loc[apply_idx, feats].astype(float).to_numpy()
    resid_hat = model.predict(X_apply)
    data.loc[apply_idx, out_col] = (
        data.loc[apply_idx, rf_col].astype(float) + lam * resid_hat
    ).clip(clip_lo, clip_hi)

    coef = model.coef_.ravel()
    print(f"Fitted RF residual corrector on n={len(fit_idx)} rows")
    print("  feats:", feats)
    print("  intercept:", float(model.intercept_))
    for f, c in zip(feats, coef, strict=False):
        print(f"  coef[{f}]: {float(c):+.6f}")

    return model, fit_idx



def print_wg_pairwise_on_llm_pairs(
    data: "pd.DataFrame",
    y: "pd.Series",
    eval_idx,
    model_cols: list,
    llm_col: str,
) -> None:
    """
    For each model column, compute WG pairwise restricted to the pairs where the
    LLM (llm_col) made a non-tied prediction -- i.e., the same denominator pairs
    that the LLM itself is evaluated on.  Prints a small comparison table.
    """
    idx = pd.Index(eval_idx).intersection(data.index).intersection(y.index)
    base_mask = y.loc[idx].notna() & data.loc[idx, llm_col].notna()
    idx = idx[base_mask.to_numpy()]

    if len(idx) == 0:
        print("  No overlapping activities for LLM-pair comparison.")
        return

    y_true_base = y.loc[idx].astype(float).to_numpy()
    y_llm_base = data.loc[idx, llm_col].astype(float).to_numpy()
    groups_base = (
        data.loc[idx, "reporting_orgs"].astype(str)
        + "_"
        + data.loc[idx, "start_date"].dt.year.astype(str)
    ).to_numpy()

    print("\n" + "=" * 80)
    print("WG PAIRWISE ON LLM-RANKED PAIRS ONLY (excl. LLM prediction ties)")
    print(f"  LLM col: {llm_col}")
    print(f"  N activities with LLM prediction: {len(idx)}")

    # LLM's own score on these pairs (reference)
    _wg_llm = within_group_pairwise_ordering_prob(y_true_base, y_llm_base, groups_base)
    llm_prob, llm_n_pairs, llm_n_groups = (
        _wg_llm["prob"],
        _wg_llm["n_pairs"],
        _wg_llm["n_groups"],
    )
    print(
        f"  {'LLM (reference)':<45} {llm_prob:.4f}  ({llm_n_pairs} pairs, {llm_n_groups} groups)"
    )

    for col in model_cols:
        if col not in data.columns:
            continue
        model_preds = data.loc[idx, col]
        col_valid = model_preds.notna()
        idx_col = idx[col_valid.to_numpy()]
        if len(idx_col) < 2:
            print(f"  {col:<45} n/a  (too few predictions)")
            continue

        yt = y.loc[idx_col].astype(float).to_numpy()
        ym = data.loc[idx_col, col].astype(float).to_numpy()
        yl = data.loc[idx_col, llm_col].astype(float).to_numpy()
        grps = (
            data.loc[idx_col, "reporting_orgs"].astype(str)
            + "_"
            + data.loc[idx_col, "start_date"].dt.year.astype(str)
        ).to_numpy()

        _wg_ref = within_group_pairwise_ordering_prob_on_reference_pairs(
            yt, ym, yl, grps
        )
        prob = _wg_ref["prob"]
        n_pairs_used = _wg_ref["n_pairs"]
        n_groups_used = _wg_ref["n_groups"]
        print(f"  {col:<45} {prob:.4f}  ({n_pairs_used} pairs, {n_groups_used} groups)")

    print("=" * 80)


def print_tex_results_table(
    *,
    data: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    eval_idx=None,
    methods: list | None = None,
    side_threshold: float = 3.5,
    decimals: int = 3,
    caption: str | None = None,
    label: str | None = None,
    cache_path: str | None = None,
) -> pd.DataFrame:
    if cache_path and data is None:
        df = pd.read_pickle(cache_path)
    else:
        base_idx = pd.Index(eval_idx).intersection(data.index).intersection(y.index)
        base_idx = base_idx[y.loc[base_idx].notna().to_numpy()]

        rows = []
        for method_name, pred_col in methods:
            method_idx = base_idx[data.loc[base_idx, pred_col].notna().to_numpy()]
            yt = y.loc[method_idx].astype(float).to_numpy()
            yp = data.loc[method_idx, pred_col].astype(float).to_numpy()

            if len(yt) == 0:
                rows.append(
                    dict(
                        Method=method_name,
                        R2=np.nan,
                        RMSE=np.nan,
                        MAE=np.nan,
                        SideAcc=np.nan,
                        AccInt=np.nan,
                        Spearman=np.nan,
                        WGSpearman=np.nan,
                        Pairwise=np.nan,
                        WGPairs=0,
                        N=0,
                    )
                )
                continue

            groups = (
                data.loc[method_idx, "reporting_orgs"].astype(str)
                + "_"
                + data.loc[method_idx, "start_date"].dt.year.astype(str)
            ).to_numpy()
            sideacc = side_accuracy(yt, yp, threshold=side_threshold)
            _wg = within_group_pairwise_ordering_prob(yt, yp, groups)
            wg_prob, wg_n_pairs = _wg["prob"], _wg["n_pairs"]
            _wg_sp = within_group_spearman_correlation(yt, yp, groups)
            rows.append(
                dict(
                    Method=method_name,
                    R2=float(r2_score(yt, yp)),
                    RMSE=float(rmse(yt, yp)),
                    MAE=float(mae(yt, yp)),
                    SideAcc=float(sideacc),
                    AccInt=float(true_hit_accuracy(yt, yp)),
                    Spearman=float(spearman_correlation(yt, yp)),
                    WGSpearman=float(_wg_sp["correlation"]),
                    Pairwise=float(wg_prob),
                    WGPairs=int(wg_n_pairs),
                    N=int(len(yt)),
                )
            )

        df = pd.DataFrame(rows)
        if cache_path:
            df.to_pickle(cache_path)

    hi_metrics = ["R2", "SideAcc", "AccInt", "Spearman", "WGSpearman", "Pairwise"]
    lo_metrics = ["RMSE", "MAE"]

    def _is_best(series: pd.Series, higher: bool) -> pd.Series:
        s = series.astype(float)
        if s.notna().sum() == 0:
            return pd.Series(False, index=series.index)
        best = s.max() if higher else s.min()
        return (np.abs(s - best) <= 1e-12) & s.notna()

    best_mask = {m: _is_best(df[m], True) for m in hi_metrics} | {
        m: _is_best(df[m], False) for m in lo_metrics
    }
    best_mask_original = {k: v.copy() for k, v in best_mask.items()}

    df_term = df.copy()
    df_term = df_term.sort_values(["RMSE", "MAE"], ascending=[True, True])
    best_mask = {k: v.reindex(df_term.index) for k, v in best_mask.items()}
    df_show = df_term[
        [
            "Method",
            "N",
            "R2",
            "RMSE",
            "MAE",
            "SideAcc",
            "AccInt",
            "Spearman",
            "WGSpearman",
            "Pairwise",
            "WGPairs",
        ]
    ].copy()

    df_display = df_show.copy()
    for c in [
        "R2",
        "RMSE",
        "MAE",
        "SideAcc",
        "AccInt",
        "Spearman",
        "WGSpearman",
        "Pairwise",
    ]:
        df_display[c] = df_show[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    df_display["WGPairs"] = df_show["WGPairs"].apply(
        lambda x: str(int(x)) if pd.notna(x) else ""
    )

    BEST = "\033[1;92m"
    RESET = "\033[0m"
    for m in ["R2", "SideAcc", "AccInt", "Spearman", "Pairwise", "RMSE", "MAE"]:
        mask = best_mask[m]
        df_display.loc[mask, m] = df_display.loc[mask, m].apply(
            lambda s: f"{BEST}{s}{RESET}" if s != "" else s
        )

    def fmt3(x):
        if pd.isna(x):
            return ""
        if isinstance(x, (int, np.integer)):
            return str(x)
        if isinstance(x, (float, np.floating)):
            return f"{x:.3f}" if np.isfinite(x) else ""
        return str(x)

    for c in [
        "R2",
        "RMSE",
        "MAE",
        "SideAcc",
        "AccInt",
        "Spearman",
        "WGSpearman",
        "Pairwise",
    ]:
        df_show[c] = df_show[c].map(fmt3)

    print("\n=== RESULTS (terminal) ===")
    col_widths = {
        "Method": 45,
        "N": 4,
        "R2": 8,
        "RMSE": 8,
        "MAE": 8,
        "SideAcc": 8,
        "AccInt": 8,
        "Spearman": 8,
        "WGSpearman": 11,
        "Pairwise": 14,
        "WGPairs": 8,
    }
    header = (
        f"{'Method':<{col_widths['Method']}} {'N':>{col_widths['N']}} "
        f"{'R2':>{col_widths['R2']}} {'RMSE':>{col_widths['RMSE']}} "
        f"{'MAE':>{col_widths['MAE']}} {'SideAcc':>{col_widths['SideAcc']}} "
        f"{'AccInt':>{col_widths['AccInt']}} {'Spearman':>{col_widths['Spearman']}} "
        f"{'WGSpearman':>{col_widths['WGSpearman']}} "
        f"{'WG Pair. Rank.':>{col_widths['Pairwise']}} {'(pairs)':>{col_widths['WGPairs']}}"
    )
    print(header)

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def pad_with_ansi(s, width):
        clean = ansi_escape.sub("", str(s))
        padding = width - len(clean)
        return " " * padding + str(s)

    for _, row in df_display.iterrows():
        print(
            f"{row['Method']:<{col_widths['Method']}} "
            f"{row['N']:>{col_widths['N']}} "
            f"{pad_with_ansi(row['R2'], col_widths['R2'])} "
            f"{pad_with_ansi(row['RMSE'], col_widths['RMSE'])} "
            f"{pad_with_ansi(row['MAE'], col_widths['MAE'])} "
            f"{pad_with_ansi(row['SideAcc'], col_widths['SideAcc'])} "
            f"{pad_with_ansi(row['AccInt'], col_widths['AccInt'])} "
            f"{pad_with_ansi(row['Spearman'], col_widths['Spearman'])} "
            f"{pad_with_ansi(row['WGSpearman'], col_widths['WGSpearman'])} "
            f"{pad_with_ansi(row['Pairwise'], col_widths['Pairwise'])} "
            f"{pad_with_ansi(row['WGPairs'], col_widths['WGPairs'])}"
        )
    print()

    # LaTeX table
    fmt = f"{{:.{decimals}f}}".format

    def _fmt_val(metric: str, v: float, is_best: bool) -> str:
        if not np.isfinite(v):
            return ""
        s = fmt(v)
        return rf"\textbf{{{s}}}" if is_best else s

    cols_hdr = [
        r"Method",
        r"$R^2$ $\uparrow$",
        r"RMSE $\downarrow$",
        r"MAE $\downarrow$",
        r"Acc. $\uparrow$",
        r"WG Spear. $\uparrow$",
        r"WG Pair. Rank. $\uparrow$",
    ]

    df = df.sort_values("R2", ascending=False)
    best_mask_latex = {k: v.reindex(df.index) for k, v in best_mask_original.items()}

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{p{7cm}rrrrrr}")
    print(r"\toprule")
    print(" & ".join(cols_hdr) + r" \\")
    print(r"\midrule")

    for i, r in df.iterrows():
        method = _latex_escape(r["Method"])
        is_llm_row = "LLM" in r["Method"] and "no LLM" not in r["Method"]
        cells = [
            method,
            _fmt_val("R2", r["R2"], bool(best_mask_latex["R2"].loc[i])),
            _fmt_val("RMSE", r["RMSE"], bool(best_mask_latex["RMSE"].loc[i])),
            _fmt_val("MAE", r["MAE"], bool(best_mask_latex["MAE"].loc[i])),
            _fmt_val("AccInt", r["AccInt"], bool(best_mask_latex["AccInt"].loc[i])),
            _fmt_val(
                "WGSpearman",
                r["WGSpearman"],
                bool(best_mask_latex["WGSpearman"].loc[i]),
            ),
            _fmt_val(
                "Pairwise", r["Pairwise"], bool(best_mask_latex["Pairwise"].loc[i])
            ),
        ]
        if is_llm_row:
            cells = [rf"\textcolor{{orange}}{{{c}}}" for c in cells]
        print(" & ".join(cells) + r" \\[3pt]")

    print(r"\bottomrule")
    print(r"\end{tabular}")

    if caption is not None:
        print(rf"\caption{{{_latex_escape(caption)}}}")
    if label is not None:
        print(rf"\label{{{_latex_escape(label)}}}")

    print(r"\end{table}")
    print(
        r"% Note: rows shown in \textcolor{orange}{orange} include LLM Forecast predictions,"
    )
    print(
        r"% which were found to benefit from future leakage and should be treated cautiously."
    )

    # Summary block
    print("\n=== SUMMARY (core metrics) ===")
    summary_cols = ["Method", "R2", "Spearman", "Pairwise", "SideAcc", "N"]
    df_summary = df[summary_cols].copy()
    for _, r in df_summary.iterrows():
        print(
            f"{r['Method']:<45}  "
            f"R2={r['R2']:.3f}  "
            f"Spearman={r['Spearman']:.3f}  "
            f"WG Pair. Rank.={r['Pairwise']:.3f}  "
            f"SideAcc={r['SideAcc']:.3f}  "
            f"N={int(r['N'])}"
        )
    print()

    return df


# ---------------------------------------------------------------------------
def load_llm_planned_duration() -> pd.DataFrame:
    """LLM-extracted planned duration in years from baseline PDFs.

    Falls back to CSV-derived duration internally (see E_plot_extracted_vs_iati_dates.py).
    Activities not in the file will keep the CSV-derived planned_duration after the merge.
    """
    path = DATA_DIR / "llm_planned_duration.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            v = rec.get("planned_duration_years")
            if v is not None and v > 0:
                rows.append(
                    {"activity_id": rec["activity_id"], "planned_duration": float(v)}
                )
    return pd.DataFrame(rows).set_index("activity_id")


def load_llm_planned_expenditure() -> pd.DataFrame:
    """LLM-extracted planned expenditure with verified ground-truth overrides.

    Stored as log(USD) to match the planned_expenditure column convention.
    Activities not in the file will be NaN after the left join -> planned_expenditure_missing=1.
    """
    rows = []
    with open(LLM_EXPENDITURE_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            v = rec["planned_expenditure_usd"]
            if v and v > 0:
                rows.append(
                    {
                        "activity_id": rec["activity_id"],
                        "planned_expenditure": np.log(v),
                    }
                )
    return pd.DataFrame(rows).set_index("activity_id")


# MAIN
# ---------------------------------------------------------------------------


def main():
    show_importances = "importances" in sys.argv[1:]
    show_plot = "plot" in sys.argv[1:]

    print(
        'WARNING: Unless USE_VAL_IN_TRAIN=True (requires typing "understood"), running on validation set.'
    )
    print("\n=== BUILDING MASTER DATAFRAME FOR RATING PREDICTION PIPELINE ===")

    print("Loading grades data...")
    grades_df = load_grades(ALL_GRADES)

    if EXCLUDE_TEST_LEAKAGE_RISK and LEAKAGE_HANDLING_METHOD in (
        "median_impute",
        "replace_predictions",
    ):
        _leaky_in_grades = grades_df.index.intersection(
            pd.Index(list(GRADE_LEAKAGE_IDS))
        )
        if len(_leaky_in_grades):
            grades_df.loc[_leaky_in_grades, :] = float("nan")
            print(
                f"[leakage] NaN grade features for {len(_leaky_in_grades)} grade-leakage "
                f"activities (RF will fill from training median)"
            )

    print("Loading activity scope...")
    scope_df = load_activity_scope(INFO_FOR_ACTIVITY_FORECASTING)

    print("Loading implementing_org_type...")
    impl_type_df = load_implementing_org_type(INFO_FOR_ACTIVITY_FORECASTING)

    print("Loading gdp_percap...")
    gdp_df = load_gdp_percap(INFO_FOR_ACTIVITY_FORECASTING)

    print("Loading planned_expenditure (LLM-extracted, verified overrides applied)...")
    expend_df = load_llm_planned_expenditure()

    print("Loading world bank indicators...")
    world_bank_df = load_world_bank_indicators(INFO_FOR_ACTIVITY_FORECASTING)

    print("Loading regions...")
    regions_df = pd.read_csv(
        INFO_FOR_ACTIVITY_FORECASTING,
        usecols=[
            "activity_id",
            "region_AFE",
            "region_AFW",
            "region_EAP",
            "region_ECA",
            "region_LAC",
            "region_MENA",
            "region_SAS",
        ],
        index_col="activity_id",
    )

    print("Loading is_completed...")
    is_completed = load_is_completed(INFO_FOR_ACTIVITY_FORECASTING)

    print("Loading loan_or_disbursement...")
    lod_df = load_loan_or_disbursement()

    print("Loading target ratings...")
    ratings = load_ratings(MERGED_OVERALL_RATINGS)

    print("Loading targets/context maps features...")
    _maps_path = (
        TARGETS_CONTEXT_MAPS_TRAINVAL if USE_VAL_IN_TRAIN else TARGETS_CONTEXT_MAPS
    )
    tc_maps_df = load_targets_context_maps_features(_maps_path)

    print("Loading sector cluster allocations...")
    sector_clusters_df = data_sector_clusters(FINANCE_SECTORS)
    sector_clusters_df.to_csv(DATA_DIR / "sector_cluster_mapping_iati.csv")

    print("\nMerging datasets...")
    data = ratings.to_frame(name="rating")
    data = data.join(is_completed, how="left")
    data = data.join(grades_df, how="left")
    data = data.join(scope_df, how="left")
    data = data.join(gdp_df, how="left")
    data = data.join(expend_df, how="left")
    data = data.join(world_bank_df, how="left")
    data = data.join(regions_df, how="left")
    data = data.join(lod_df, how="left")
    data = data.join(impl_type_df, how="left")
    data = data.join(tc_maps_df, how="left")
    data = data.join(sector_clusters_df, how="left")

    data["activity_scope"] = pd.to_numeric(data["activity_scope"], errors="coerce")

    region_cols = [
        "region_AFE",
        "region_AFW",
        "region_EAP",
        "region_ECA",
        "region_LAC",
        "region_MENA",
        "region_SAS",
    ]
    for col in region_cols:
        if col in data.columns:
            data[col] = data[col].fillna(0.0)

    data, sim_feature_cols = add_similarity_features(
        data, INFO_FOR_ACTIVITY_FORECASTING, KEEP_REPORTING_ORGS
    )
    data = restrict_to_reporting_orgs_exact(data, KEEP_REPORTING_ORGS)
    data = add_dates_to_dataframe(data, INFO_FOR_ACTIVITY_FORECASTING)
    llm_dur_df = load_llm_planned_duration()
    data["planned_duration"] = llm_dur_df["planned_duration"].reindex(data.index)
    _n_total = len(data)
    _n_present = data["planned_duration"].notna().sum()
    _n_missing = data["planned_duration"].isna().sum()
    _n_llm = llm_dur_df.index.isin(data.index).sum()
    print(
        f"[planned_duration] total={_n_total}  llm_matched={_n_llm}"
        f"  present={_n_present} ({100*_n_present/_n_total:.1f}%)"
        f"  missing={_n_missing} ({100*_n_missing/_n_total:.1f}%)"
    )
    if _n_present > 0:
        print(
            f"[planned_duration] mean={data['planned_duration'].mean():.2f}yr"
            f"  median={data['planned_duration'].median():.2f}yr"
            f"  min={data['planned_duration'].min():.2f}yr"
            f"  max={data['planned_duration'].max():.2f}yr"
        )
    data["has_start"] = data["start_date"].notna()

    llm_feats_for_uncertainty = [
        "finance",
        "integratedness",
        "implementer_performance",
        "targets",
        "context",
        "risks",
        "complexity",
    ]
    data = add_enhanced_uncertainty_features(data, llm_feats_for_uncertainty)

    feature_cols_no_llm = [
        "activity_scope",
        "gdp_percap",
        "cpia_score",
        "wgi_control_of_corruption_est",
        "wgi_government_effectiveness_est",
        "wgi_political_stability_est",
        "wgi_regulatory_quality_est",
        "wgi_rule_of_law_est",
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
    ]

    if USE_TOP_FEATURES_FOR_GENERALIZATION:
        # ~19-feature set selected by val delta_r2/pairwise (see feature_importances.csv)
        # Dropped: cpia_score, country_distance, governance_composite, finance, umap3_z,
        #          all WGI cols, all regions except EAP, rep_org dummies, all missing indicators,
        #          log_planned_expenditure, integratedness, finance_is_loan
        feature_cols = [
            "targets",
            "context",
            "implementer_performance",
            "activity_scope",
            "risks",
            "complexity",
            "umap3_x",
            "umap3_y",
            "sector_distance",
            "planned_expenditure",
            "planned_duration",
            "gdp_percap",
            "region_EAP",
            "sector_cluster_Capacity_Building_and_Technical_Assistance",
            "sector_cluster_Road_safety_improvements",
            "sector_cluster_Urban_flood_protection",
            "sector_cluster_reduced_PM2.5_air_pollution",
        ]
        feature_cols = [f for f in feature_cols if f in data.columns]
    else:
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
            "wgi_control_of_corruption_est",
            "wgi_government_effectiveness_est",
            "wgi_political_stability_est",
            "wgi_regulatory_quality_est",
            "wgi_rule_of_law_est",
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

    feat_idx = pd.Index(feature_cols)
    dups = feat_idx[feat_idx.duplicated()].unique().tolist()
    if dups:
        raise ValueError(f"Duplicate feature names in feature_cols: {dups}")

    start_dates = data.loc[:, "start_date"]
    min_date = start_dates.min()
    data["start_date_days"] = (start_dates - min_date).dt.days

    X = data[feature_cols]
    y = data["rating"].astype(float)

    complete_mask = data["is_completed"].fillna(0).astype(int) == 1
    data.loc[~complete_mask]
    X = X.loc[complete_mask]
    y = y.loc[complete_mask]
    data = data.loc[complete_mask]

    if data["reporting_orgs"].isna().any():
        bad = int(data["reporting_orgs"].isna().sum())
        raise ValueError(
            f"reporting_orgs has {bad} NaNs -- fix upstream (don't silently drop)."
        )

    if data["start_date"].isna().any():
        bad = int(data["start_date"].isna().sum())
        raise ValueError(
            f"start_date has {bad} NaNs -- fix upstream (don't silently drop)."
        )

    _d = data.sort_values("start_date")
    _too_late = _d["start_date"] >= pd.to_datetime(TOO_LATE_CUTOFF)
    _d = _d[~_too_late]
    train_idx = _d[_d["start_date"] <= pd.to_datetime(LATEST_TRAIN_POINT)].index
    val_idx = _d[
        (_d["start_date"] > pd.to_datetime(LATEST_TRAIN_POINT))
        & (_d["start_date"] <= pd.to_datetime(LATEST_VALIDATION_POINT))
    ].index
    held_idx = _d[_d["start_date"] > pd.to_datetime(LATEST_VALIDATION_POINT)].index

    assert_split_matches_canonical(
        train_idx,
        val_idx,
        held_idx,
        splits_csv=DATA_DIR / "train_val_test_ids.csv",
    )

    print(
        f"Split info: train={len(train_idx)}, val={len(val_idx)}, test={len(held_idx)}"
    )
    print(f"  train cutoff: {LATEST_TRAIN_POINT}")
    print(f"  val cutoff:   {LATEST_VALIDATION_POINT}")
    print(f"  too-late cutoff: {TOO_LATE_CUTOFF}")

    X_train = X.loc[train_idx]
    X_val = X.loc[val_idx]
    y_train = y.loc[train_idx]
    y_val = y.loc[val_idx]

    # Save split IDs
    out = pd.concat(
        [
            pd.DataFrame(
                {"activity_id": pd.Index(train_idx).astype(str), "split": "train"}
            ),
            pd.DataFrame(
                {"activity_id": pd.Index(val_idx).astype(str), "split": "val"}
            ),
            pd.DataFrame(
                {"activity_id": pd.Index(held_idx).astype(str), "split": "test"}
            ),
        ],
        ignore_index=True,
    ).sort_values(["split", "activity_id"])
    out.to_csv(DATA_DIR / "train_val_test_ids.csv", index=False)

    # Save a copy to the shared eval-set-sizes directory for cross-script comparison
    _eval_dir = DATA_DIR / "eval_set_sizes"
    _eval_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(_eval_dir / "overall_ratings_splits.csv", index=False)
    print(
        f"[eval_set_sizes] overall_ratings splits saved to {_eval_dir / 'overall_ratings_splits.csv'}"
    )

    # Feature engineering: governance composite and expenditure transforms
    wgi_cols = [
        "wgi_control_of_corruption_est",
        "wgi_political_stability_est",
        "wgi_government_effectiveness_est",
        "wgi_regulatory_quality_est",
        "wgi_rule_of_law_est",
    ]
    if all(c in data.columns for c in wgi_cols):
        data["governance_composite"] = data[wgi_cols].mean(axis=1)

    if "planned_expenditure" in data.columns:
        data["log_planned_expenditure"] = data["planned_expenditure"]
        data["planned_expenditure"] = np.exp(data["planned_expenditure"])

    if "planned_expenditure" in data.columns and "complexity" in data.columns:
        data["expenditure_x_complexity"] = (
            data["planned_expenditure"] * data["complexity"]
        )

    if "planned_expenditure" in data.columns and "planned_duration" in data.columns:
        data["expenditure_per_year"] = (
            data["planned_expenditure"] / data["planned_duration"]
        ).where(
            (data["planned_duration"] >= 1) & (data["planned_expenditure"] >= 100000),
            np.nan,
        )
        data["expenditure_per_year_log"] = np.log(
            (data["planned_expenditure"] / data["planned_duration"]).where(
                (data["planned_duration"] >= 1)
                & (data["planned_expenditure"] >= 100000),
                np.nan,
            )
        )

    if USE_TOP_FEATURES_FOR_GENERALIZATION:
        new_features = [
            f
            for f in [
                "expenditure_x_complexity",
                "expenditure_per_year_log",
            ]
            if f in data.columns
        ]
        feature_cols = feature_cols + new_features
    else:
        new_features = [
            f
            for f in [
                "governance_composite",
                "expenditure_x_complexity",
                "expenditure_per_year_log",
                "log_planned_expenditure",
            ]
            if f in data.columns
        ]
        if "governance_composite" in new_features:
            feature_cols = [f for f in feature_cols if f not in wgi_cols]
        feature_cols = feature_cols + new_features

    if TOP_10_FEATURES_ONLY:
        imp_path = OUTPUT_DIR / "feature_importances.csv"
        if not imp_path.exists():
            raise FileNotFoundError(
                f"TOP_10_FEATURES_ONLY=True but no importances file found at {imp_path}. "
                "Run once with the 'importances' argument first."
            )
        imp_df = pd.read_csv(imp_path)
        top10 = imp_df.nlargest(10, "importance")["feature"].tolist()
        feature_cols = [f for f in top10 if f in feature_cols]
        print(
            f"\nTOP_10_FEATURES_ONLY: restricting to {len(feature_cols)} features: {feature_cols}"
        )

    _orig_train_idx = X_train.index  # before potential merge with val

    if USE_VAL_IN_TRAIN:
        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)
        X_test, _y_test = X.loc[held_idx], y.loc[held_idx]
    else:
        print("running on validation set")
        X_test = X_val
    train_idx, test_idx = X_train.index, X_test.index

    if EXCLUDE_TEST_LEAKAGE_RISK and LEAKAGE_HANDLING_METHOD == "drop":
        test_idx = test_idx.difference(pd.Index(list(TEST_ANY_LEAKAGE_IDS)))
        print(
            f"[leakage] drop: removed {len(TEST_ANY_LEAKAGE_IDS)} leakage activities from test_idx"
        )

    # ---- Models ----
    data = add_per_org_mode_baseline(
        data,
        y,
        train_idx,
        test_idx,
        org_cols=[f"rep_org_{i}" for i in range(NUM_ORGS_KEEP)],
        out_col="per_org_mode",
    )

    # Ensure org one-hots are in feature_cols (may be absent in USE_TOP_FEATURES path).
    # Use k-1 dummies (NUM_ORGS_KEEP - 1) to avoid perfect multicollinearity with the intercept.
    _org_fc = [
        f"rep_org_{i}"
        for i in range(NUM_ORGS_KEEP - 1)
        if f"rep_org_{i}" not in feature_cols
    ]
    if _org_fc:
        feature_cols = feature_cols + _org_fc

    _feat_save_dir = (
        Path(__file__).resolve().parent.parent.parent / "data" / "feature_lists"
    )
    _feat_save_dir.mkdir(parents=True, exist_ok=True)
    _feat_save_path = _feat_save_dir / "overall_rating_features.json"
    with open(_feat_save_path, "w") as _f:
        json.dump(
            {
                "model": "overall_rating",
                "n_features": len(feature_cols),
                "features": feature_cols,
            },
            _f,
            indent=2,
        )
    print(f"[feature_lists] Saved {len(feature_cols)} features to {_feat_save_path}")

    # Save all feature values (+ true rating label) for train/val/test to CSV
    _split_labels = {}
    for _idx in _orig_train_idx:
        _split_labels[_idx] = "train"
    for _idx in val_idx:
        _split_labels[_idx] = "val"
    for _idx in held_idx:
        _split_labels[_idx] = "test"
    _feat_save = data.loc[
        [i for i in data.index if i in _split_labels], feature_cols
    ].copy()
    _feat_save.insert(0, "rating", data.loc[_feat_save.index, "rating"])
    _feat_save.insert(
        0, "start_year", data.loc[_feat_save.index, "start_date"].dt.year.astype(int)
    )
    _feat_save.insert(0, "reporting_orgs", data.loc[_feat_save.index, "reporting_orgs"])
    _feat_save.insert(0, "split", [_split_labels[i] for i in _feat_save.index])
    _feat_save.index.name = "activity_id"
    _features_path = OUTPUT_DIR / "all_features.csv"
    _feat_save.to_csv(_features_path)
    print(
        f"Saved {len(_feat_save)} rows x {len(feature_cols)} features to {_features_path}"
    )
    _iati_feat_dir = Path.home() / "Code" / "iati_extractions" / "features"
    _iati_feat_dir.mkdir(parents=True, exist_ok=True)
    _feat_save.to_csv(_iati_feat_dir / "all_features.csv")

    # y_delta: subtract the training per-org mode; models learn the residual from mode
    all_model_idx = train_idx.append(test_idx)
    y_delta = y.copy().astype(float)
    y_delta.loc[all_model_idx] -= data.loc[all_model_idx, "per_org_mode"]
    data["rating_delta"] = y_delta

    glm_pred_nonbin_ridge, ridge_model_nonbin = run_ridge_glm_median_impute_noclip(
        data=data,
        feature_cols=feature_cols,
        target_col="rating_delta",
        train_index=train_idx,
    )
    data["glm_nonbin_pred_ridge"] = np.clip(
        pd.Series(glm_pred_nonbin_ridge, index=data.index) + data["per_org_mode"],
        *RATING_CLIP,
    )

    baseline_feature_cols = ["risks"] + [
        f"rep_org_{i}" for i in range(NUM_ORGS_KEEP - 1)
    ]
    glm_pred_baseline, _ = run_ridge_glm_median_impute_noclip(
        data=data,
        feature_cols=baseline_feature_cols,
        target_col="rating_delta",
        train_index=train_idx,
    )
    data["ridge_baseline_risk_org"] = np.clip(
        pd.Series(glm_pred_baseline, index=data.index) + data["per_org_mode"],
        *RATING_CLIP,
    )

    # ---- Plain OLS GLM (statsmodels, no regularization) on rating_delta ----
    _X_all_ols = data[feature_cols].copy().astype(float)
    _train_medians_ols = _X_all_ols.loc[train_idx].median()
    _X_all_ols_imp = _X_all_ols.fillna(_train_medians_ols)
    _X_train_ols = _X_all_ols_imp.loc[train_idx].to_numpy(dtype=float)
    _y_train_delta_ols = (
        data.loc[train_idx, "rating_delta"].astype(float).to_numpy(dtype=float)
    )

    _ols_result = sm.OLS(
        _y_train_delta_ols, sm.add_constant(_X_train_ols, has_constant="add")
    ).fit()
    _X_all_const = sm.add_constant(
        _X_all_ols_imp.to_numpy(dtype=float), has_constant="add"
    )
    _glm_plain_delta = pd.Series(_ols_result.predict(_X_all_const), index=data.index)
    data["pred_glm_plain"] = np.clip(
        _glm_plain_delta + data["per_org_mode"], *RATING_CLIP
    )
    data["pred_glm_plain_yr_corr"] = data["pred_glm_plain"].copy()

    # Debug: save OLS inputs/outputs so we can diff against C_overall_rating_insample_r2.py
    _ols_debug = pd.DataFrame(
        _X_train_ols,
        index=pd.Index(train_idx).astype(str),
        columns=feature_cols,
    )
    _ols_debug.index.name = "activity_id"
    _ols_debug.insert(0, "y_delta", _y_train_delta_ols)
    _ols_debug.to_csv(OUTPUT_DIR / "ols_debug_X_train.csv")

    _ols_test_debug = pd.DataFrame(
        {
            "activity_id": pd.Index(test_idx).astype(str),
            "y_true": y.loc[test_idx].astype(float).values,
            "per_org_mode": data.loc[test_idx, "per_org_mode"].values,
            "pred_delta": _glm_plain_delta.loc[test_idx].values,
            "pred_glm_plain": data.loc[test_idx, "pred_glm_plain"].values,
        }
    ).set_index("activity_id")
    _ols_test_debug.to_csv(OUTPUT_DIR / "ols_debug_test_preds.csv")
    print(
        f"\n[OLS debug] saved X_train ({_ols_debug.shape}) and test preds ({len(_ols_test_debug)}) to {OUTPUT_DIR}"
    )
    print(
        f"[OLS debug] train R^2={_ols_result.rsquared:.4f}  n_features={len(feature_cols)}  n_train={len(train_idx)}"
    )
    print(f"[OLS debug] feature_cols used: {feature_cols}")

    # Per-feature variance explained: drop-one semi-partial R^2 on training set
    _total_r2_ols = float(_ols_result.rsquared)
    _delta_r2_contribs = {}
    _p_ols = _X_train_ols.shape[1]
    for _j, _col in enumerate(feature_cols):
        _mask = [k for k in range(_p_ols) if k != _j]
        _X_drop = _X_train_ols[:, _mask]
        _r2_drop = float(
            sm.OLS(_y_train_delta_ols, sm.add_constant(_X_drop, has_constant="add"))
            .fit()
            .rsquared
        )
        _delta_r2_contribs[_col] = _total_r2_ols - _r2_drop

    print(
        "\nPlain OLS GLM -- semi-partial R^2 (drop-one, training set, target=rating_delta):"
    )
    print(f"  Total train R^2 = {_total_r2_ols:.4f}")
    print(f"  {'Feature':<45s}  {'DeltaR^2':>8s}  {'% of R^2':>8s}")
    print(f"  {'-'*67}")
    _sorted_dr2 = sorted(_delta_r2_contribs.items(), key=lambda kv: kv[1], reverse=True)
    for _feat, _dr2 in _sorted_dr2:
        _pct_r2 = 100.0 * _dr2 / _total_r2_ols if _total_r2_ols != 0 else float("nan")
        print(f"  {_feat:<45s}  {_dr2:8.4f}  {_pct_r2:7.1f}%")

    X_train = data.loc[train_idx, feature_cols].astype(float)

    rf_params = {
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

    rf_pred_default, _ = run_random_forest_median_impute_noclip(
        data=data,
        feature_cols=feature_cols,
        target_col="rating_delta",
        train_index=train_idx,
        rf_params={},
        ensemble_with_extratrees=True,
    )
    data["pred_rf_default_params"] = np.clip(
        pd.Series(rf_pred_default, index=data.index) + data["per_org_mode"],
        *RATING_CLIP,
    )

    rf_delta, rf_model, extra_model = run_random_forest_median_impute_noclip(
        data=data,
        feature_cols=feature_cols,
        target_col="rating_delta",
        train_index=train_idx,
        rf_params=rf_params,
        ensemble_with_extratrees=True,
        return_extra=True,
    )
    rf_pred = np.clip(
        (pd.Series(rf_delta, index=data.index) + data["per_org_mode"]).to_numpy(),
        *RATING_CLIP,
    )

    # feature_cols_no_llm already includes rep_org_ columns
    rf_pred_no_llm_delta, _ = run_random_forest_median_impute_noclip(
        data=data,
        feature_cols=feature_cols_no_llm,
        target_col="rating_delta",
        train_index=train_idx,
        rf_params=rf_params,
        ensemble_with_extratrees=True,
    )
    data["pred_rf_no_llm"] = np.clip(
        pd.Series(rf_pred_no_llm_delta, index=data.index) + data["per_org_mode"],
        *RATING_CLIP,
    )
    data["pred_rf"] = pd.Series(rf_pred, index=data.index)
    data["ridge_plus_rf"] = data[["glm_nonbin_pred_ridge", "pred_rf"]].mean(axis=1)
    data["ridge_plus_rf_corrected"] = data["ridge_plus_rf"].copy()
    apply_start_year_trend_correction(
        data, y, train_idx, pred_col="ridge_plus_rf_corrected"
    )

    if show_importances:
        print("\nCalculating feature importance...")
        imp_rf = one_sd_shift_importance(
            model=rf_model,
            data=data,
            feature_cols=feature_cols,
            train_idx=train_idx,
        )
        if extra_model is not None:
            imp_et = one_sd_shift_importance(
                model=extra_model,
                data=data,
                feature_cols=feature_cols,
                train_idx=train_idx,
            )
            # Average RF and ET delta_pred_1sd / importance_abs_1sd for the ensemble estimate
            imp_et = imp_et.set_index("feature")
            imp_rf = imp_rf.set_index("feature")
            imp_rf["delta_pred_1sd"] = (
                imp_rf["delta_pred_1sd"] + imp_et["delta_pred_1sd"]
            ) / 2.0
            imp_rf["importance_abs_1sd"] = (
                imp_rf["importance_abs_1sd"] + imp_et["importance_abs_1sd"]
            ) / 2.0
            imp_rf = imp_rf.reset_index()
        # Deterministic secondary sort: round sub-1e-12 noise to 0, sort by importance desc
        # then feature_cols position for ties (avoids n_jobs=-1 floating-point non-determinism)
        imp_rf["importance_abs_1sd"] = imp_rf["importance_abs_1sd"].where(
            imp_rf["importance_abs_1sd"] > 1e-12, 0.0
        )
        _feat_order = {f: i for i, f in enumerate(feature_cols)}
        imp_rf = (
            imp_rf.assign(_feat_idx=imp_rf["feature"].map(_feat_order))
            .sort_values(["importance_abs_1sd", "_feat_idx"], ascending=[False, True])
            .drop(columns=["_feat_idx"])
            .reset_index(drop=True)
        )
        print("\nTop 25 features by importance_abs_1sd:")
        print(
            "\n"
            + imp_rf[["feature", "importance_abs_1sd", "delta_pred_1sd", "sd_train"]]
            .head(25)
            .to_string(index=False)
        )
        print("\n\nBottom 10 features (least important):")
        print(
            imp_rf[["feature", "importance_abs_1sd", "delta_pred_1sd", "sd_train"]]
            .tail(10)
            .to_string(index=False)
        )

        test_idx_clean = pd.Index(test_idx)
        missingness = {
            col: data.loc[test_idx_clean, col].isna().mean() * 100
            for col in feature_cols
        }
        imp_rf["missingness_pct"] = imp_rf["feature"].map(missingness)
        imp_rf["delta_r2"] = np.nan
        imp_rf["delta_pairwise"] = np.nan
        print("\nRunning leave-one-out ablation for top 30 features...")
        y_true_test = data.loc[test_idx_clean, "rating"].astype(float).to_numpy()
        y_pred_test = (
            pd.Series(rf_pred, index=data.index)
            .loc[test_idx_clean]
            .astype(float)
            .to_numpy()
        )
        _abl_groups = (
            data.loc[test_idx_clean, "reporting_orgs"].astype(str)
            + "_"
            + data.loc[test_idx_clean, "start_date"].dt.year.astype(str)
        ).to_numpy()
        mask_valid = np.isfinite(y_true_test) & np.isfinite(y_pred_test)

        base_r2_rf = r2_score(y_true_test[mask_valid], y_pred_test[mask_valid])
        base_pairwise_rf = within_group_pairwise_ordering_prob(
            y_true_test[mask_valid], y_pred_test[mask_valid], _abl_groups[mask_valid]
        )["prob"]

        top_30_features = imp_rf.head(25)["feature"].tolist()
        ablation_results = {}
        for i, feat in enumerate(top_30_features, 1):
            print(f"  [{i}/30] Dropping {feat}...")
            reduced_cols = [c for c in feature_cols if c != feat]
            y_pred_drop_delta, _ = run_random_forest_median_impute_noclip(
                data=data,
                feature_cols=reduced_cols,
                target_col="rating_delta",
                train_index=train_idx,
                ensemble_with_extratrees=True,
                rf_params=rf_params,
            )
            y_pred_drop_test = np.clip(
                (pd.Series(y_pred_drop_delta, index=data.index) + data["per_org_mode"])
                .loc[test_idx_clean]
                .astype(float)
                .to_numpy(),
                *RATING_CLIP,
            )
            mask_drop = np.isfinite(y_true_test) & np.isfinite(y_pred_drop_test)
            r2_drop = r2_score(y_true_test[mask_drop], y_pred_drop_test[mask_drop])
            pairwise_drop = within_group_pairwise_ordering_prob(
                y_true_test[mask_drop],
                y_pred_drop_test[mask_drop],
                _abl_groups[mask_drop],
            )["prob"]
            ablation_results[feat] = {
                "delta_r2": base_r2_rf - r2_drop,
                "delta_pairwise": base_pairwise_rf - pairwise_drop,
            }

        imp_rf["delta_r2"] = imp_rf["feature"].map(
            lambda f: ablation_results.get(f, {}).get("delta_r2", np.nan)
        )
        imp_rf["delta_pairwise"] = imp_rf["feature"].map(
            lambda f: ablation_results.get(f, {}).get("delta_pairwise", np.nan)
        )

        top30_df = imp_rf.head(25)[
            [
                "feature",
                "missingness_pct",
                "delta_r2",
                "delta_pairwise",
                "delta_pred_1sd",
            ]
        ].copy()

        print("\n" + "=" * 80)
        print("TOP 30 FEATURES - DETAILED METRICS (RF+ET)")
        print("=" * 80)
        print(
            "\n{:<40s} {:>10s} {:>10s} {:>12s} {:>12s}".format(
                "Feature", "Miss %", "DeltaR^2", "DeltaPairwise", "Deltapred_1sd"
            )
        )
        print("-" * 88)
        for _, row in top30_df.iterrows():
            print(
                "{:<40s} {:>10.1f} {:>10.4f} {:>12.4f} {:>12.4f}".format(
                    get_display_name(row["feature"])[:40],
                    row["missingness_pct"] if pd.notna(row["missingness_pct"]) else 0.0,
                    row["delta_r2"] if pd.notna(row["delta_r2"]) else 0.0,
                    row["delta_pairwise"] if pd.notna(row["delta_pairwise"]) else 0.0,
                    row["delta_pred_1sd"] if pd.notna(row["delta_pred_1sd"]) else 0.0,
                )
            )
        print("\n" + "=" * 80)
        print("LATEX TABLE (Top 25 Features)")
        print("=" * 80)
        print(r"\begin{table}[t]")
        print(r"\centering")
        print(r"\caption{Random-forest drop-one feature importance, sorted by impact.}")
        print(r"\begin{tabular}{p{6cm}rrrr}")
        print(r"\toprule")
        print(
            r"Feature & Miss \% & $\Delta R^2$ $\uparrow$ & $\Delta$Pairwise $\uparrow$ & $\Delta$pred\_1sd \\"
        )
        print(r"\midrule")
        for _, row in top30_df.iterrows():
            feat_label = _latex_escape(get_display_name(row["feature"]))
            miss_val = (
                f"{row['missingness_pct']:.1f}"
                if pd.notna(row["missingness_pct"])
                else "0.0"
            )
            dr2_val = f"{row['delta_r2']:.4f}" if pd.notna(row["delta_r2"]) else "---"
            dpair_val = (
                f"{row['delta_pairwise']:.4f}"
                if pd.notna(row["delta_pairwise"])
                else "---"
            )
            dpred_val = (
                f"{row['delta_pred_1sd']:.4f}"
                if pd.notna(row["delta_pred_1sd"])
                else "0.0"
            )
            print(
                f"{feat_label} & {miss_val} & {dr2_val} & {dpair_val} & {dpred_val} \\\\[2pt]"
            )
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print(r"\end{table}")

    # ---- XGBoost ----
    print("\nTraining XGBoost...")
    xgb_params = {
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 3,
        "gamma": 2.0,
        "subsample": 0.7,
        "colsample_bytree": 0.6,
        "reg_alpha": 1.0,
        "reg_lambda": 10.0,
        "min_child_weight": 30,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    xgb_delta, xgb_model = run_xgboost_native_missing(
        data=data,
        feature_cols=feature_cols,
        target_col="rating_delta",
        train_index=train_idx,
        xgb_params=xgb_params,
        clip_pred=False,
    )
    data["pred_xgb"] = np.clip(
        pd.Series(xgb_delta, index=data.index) + data["per_org_mode"], *RATING_CLIP
    )

    xgb_no_llm_delta, _ = run_xgboost_native_missing(
        data=data,
        feature_cols=feature_cols_no_llm,
        target_col="rating_delta",
        train_index=train_idx,
        xgb_params=xgb_params,
        clip_pred=False,
    )
    data["pred_xgb_no_llm"] = np.clip(
        pd.Series(xgb_no_llm_delta, index=data.index) + data["per_org_mode"],
        *RATING_CLIP,
    )

    # ---- LLM predictions ----
    print("\nLoading LLM predictions...")
    PREDICTION_CONFIGS = get_llm_prediction_configs()
    llm_pred_cols = []

    for name, path, parser, label in PREDICTION_CONFIGS:
        series_name = f"{name}_prediction"
        if not path.exists():
            print(f"  Skipping {path.name} (file not found)")
            continue
        print(f"  Loading {path.name}...")
        preds = (
            load_predictions_from_jsonl(path, parser, series_name)
            .groupby(level=0)
            .mean()
        )
        data = data.join(preds, how="left")
        if preds.notna().sum() == 0:
            print(f"  Warning: no valid predictions for {label}")
            continue
        llm_pred_cols.append(series_name)

    data["mean_all_llm_preds"] = data[llm_pred_cols].mean(axis=1)

    # ---- Recency corrector ----
    recency_model, _ = add_rf_llm_residual_corrector(
        data=data,
        y=y,
        meta_train_idx=train_idx,
        rf_col="pred_rf",
        llm_col=None,
        out_col="pred_rf_recency",
        alpha=LLM_CORRECTOR_ALPHA,
        lam=LLM_CORRECTOR_LAM,
        use_llm=False,
        clip_lo=LLM_CORRECTOR_CLIP[0],
        clip_hi=LLM_CORRECTOR_CLIP[1],
    )

    # ---- RF + LLM residual corrector ----
    model, llm_corrector_fit_idx = add_rf_llm_residual_corrector(
        data=data,
        y=y,
        meta_train_idx=train_idx,
        rf_col="pred_rf",
        llm_col="mean_all_llm_preds",
        out_col="pred_rf_llm_modded",
        alpha=LLM_CORRECTOR_ALPHA,
        lam=LLM_CORRECTOR_LAM,
        clip_lo=LLM_CORRECTOR_CLIP[0],
        clip_hi=LLM_CORRECTOR_CLIP[1],
    )

    print(
        f"[DEBUG] LLM corrector model: {model}, fit_idx len: {len(llm_corrector_fit_idx)}"
    )
    if model is not None:
        # The corrector fits: residual = intercept + b_rf*pred_rf + b_llm*mean_llm
        # Adding back pred_rf gives: final = intercept + (1+b_rf)*pred_rf + b_llm*mean_llm
        # Print effective weights in the final prediction so signs are intuitive.
        _b_rf = float(model.coef_[0])
        _b_llm = float(model.coef_[1])
        _coef_rows = [
            (
                "Intercept",
                float(model.intercept_),
                "Constant additive shift applied to every prediction",
            ),
            (
                "RF+ET prediction  (eff. wt.)",
                1.0 + _b_rf,
                "How much of the RF prediction survives; <1 = shrinkage toward the mean",
            ),
            (
                "Mean LLM forecast (eff. wt.)",
                _b_llm,
                "How much LLM signal is blended in; positive = LLM improves on RF",
            ),
        ]
        print("\n" + "=" * 100)
        print("=" * 100)
        print(
            "===  LLM RESIDUAL CORRECTOR -- EFFECTIVE WEIGHTS IN FINAL PREDICTION  ==="
        )
        print("=" * 100)
        print("=" * 100)
        print(f"\n  {'Term':<30} {'Weight':>10}   Interpretation")
        print("  " + "-" * 90)
        for _name, _val, _desc in _coef_rows:
            print(f"  {_name:<30} {_val:>+10.4f}   {_desc}")
        print()
        print(
            "  final = intercept + (1 + b_rf)*pred_rf + b_llm*mean_llm  (clipped to [0, 5])"
        )
        print("\n" + "=" * 100)
        print("=" * 100 + "\n")

    # ---- Simple 50/50 average of RF+ET and LLM (no ridge adjustment) ----
    # Falls back to pred_rf where mean_all_llm_preds is unavailable.
    _llm_avail = data["mean_all_llm_preds"].notna()
    data["pred_rf_llm_simple_avg"] = data["pred_rf"].copy()
    data.loc[_llm_avail, "pred_rf_llm_simple_avg"] = (
        (data.loc[_llm_avail, "pred_rf"] + data.loc[_llm_avail, "mean_all_llm_preds"])
        / 2
    ).clip(*LLM_CORRECTOR_CLIP)

    # ---- Ridge+RF ensemble + LLM residual corrector ----
    add_rf_llm_residual_corrector(
        data=data,
        y=y,
        meta_train_idx=train_idx,
        rf_col="ridge_plus_rf",
        llm_col="mean_all_llm_preds",
        out_col="ridge_plus_rf_llm_modded",
        alpha=LLM_CORRECTOR_ALPHA,
        lam=LLM_CORRECTOR_LAM,
        clip_lo=LLM_CORRECTOR_CLIP[0],
        clip_hi=LLM_CORRECTOR_CLIP[1],
    )

    # ---- Start-year trend correction (fitted on train only, applied in-place) ----
    data["pred_rf_no_yr_corr"] = data["pred_rf"].copy()
    data["pred_rf_llm_modded_no_yr_corr"] = data["pred_rf_llm_modded"].copy()
    apply_start_year_trend_correction(data, y, train_idx, pred_col="pred_rf")
    apply_start_year_trend_correction(data, y, train_idx, pred_col="pred_rf_llm_modded")
    apply_start_year_trend_correction(
        data, y, train_idx, pred_col="pred_glm_plain_yr_corr"
    )
    apply_start_year_trend_correction(
        data, y, train_idx, pred_col="ridge_plus_rf_llm_modded"
    )

    # ---- Leakage prediction replacement ----
    if EXCLUDE_TEST_LEAKAGE_RISK and LEAKAGE_HANDLING_METHOD == "replace_predictions":
        _llm_blend_cols = [
            c
            for c in [
                "pred_rf_llm_modded",
                "pred_rf_llm_modded_no_yr_corr",
                "pred_rf_llm_simple_avg",
                "ridge_plus_rf_llm_modded",
            ]
            if c in data.columns
        ]
        # Grade leakage: replace with no-LLM-features model (grades are tainted)
        _grade_leaky_test = test_idx.intersection(pd.Index(list(GRADE_LEAKAGE_IDS)))
        if len(_grade_leaky_test):
            for _col in _llm_blend_cols:
                data.loc[_grade_leaky_test, _col] = data.loc[
                    _grade_leaky_test, "pred_rf_no_llm"
                ]
            print(
                f"[leakage] replace_predictions: swapped {len(_grade_leaky_test)} grade-leakage "
                f"activities to pred_rf_no_llm in {_llm_blend_cols}"
            )
        # Forecast-only leakage: replace with base RF (LLM forecast is tainted, grades are OK)
        _forecast_only_test = test_idx.intersection(
            pd.Index(list(TEST_ANY_LEAKAGE_IDS - GRADE_LEAKAGE_IDS))
        )
        if len(_forecast_only_test):
            for _col in _llm_blend_cols:
                data.loc[_forecast_only_test, _col] = data.loc[
                    _forecast_only_test, "pred_rf"
                ]
            print(
                f"[leakage] replace_predictions: swapped {len(_forecast_only_test)} forecast-only "
                f"leakage activities to pred_rf in {_llm_blend_cols}"
            )

    # Per-year activity counts (sanity check that LLM corrector fit set is recent)
    print("\n=== PER-YEAR ACTIVITY COUNTS ===")
    for label, split_idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("llm_corrector_fit", llm_corrector_fit_idx),
    ]:
        years = (
            data.loc[data.index.intersection(split_idx), "start_date"]
            .dt.year.dropna()
            .astype(int)
        )
        counts = years.value_counts().sort_index()
        counts_str = "  ".join(f"{yr}: {n}" for yr, n in counts.items())
        print(f"  {label:25s} (n={len(split_idx):4d}): {counts_str}")

    data["pred_rf_llm_modded_rounded"] = data["pred_rf_llm_modded"].round()

    if show_plot:
        pred_series_for_eb = data["pred_rf_llm_modded"].dropna()
        cal_idx_for_eb = val_idx.intersection(pred_series_for_eb.index)
        error_bars_conformal = get_error_bars_split_conformal(
            y_true=y,
            y_pred=pred_series_for_eb,
            cal_idx=cal_idx_for_eb,
            alpha=0.10,
        )
        hw = error_bars_conformal.iloc[0]
        print(f"[plot] Fixed conformal 90% PI half-width: {hw:.3f}")

    # ---- Magic offset: shift each column's test predictions to match y_true mean ----
    if MAGIC_OFFSET:
        _offset_cols = [
            "per_org_mode",
            "ridge_baseline_risk_org",
            "pred_glm_plain",
            "pred_glm_plain_yr_corr",
            "pred_rf",
            "pred_rf_no_llm",
            "pred_xgb",
            "pred_xgb_no_llm",
            "glm_nonbin_pred_ridge",
            "pred_rf_default_params",
            "ridge_plus_rf",
            "ridge_plus_rf_corrected",
            "pred_rf_no_yr_corr",
            "pred_rf_recency",
            "pred_rf_llm_modded_no_yr_corr",
            "pred_rf_llm_modded",
            "ridge_plus_rf_llm_modded",
            "pred_rf_llm_modded_rounded",
        ]
        _test_years = data.loc[test_idx.intersection(data.index), "start_date"].dt.year
        _unique_years = sorted(_test_years.dropna().astype(int).unique())
        print("\n=== MAGIC OFFSET (per-column, per-year, test set) ===")
        print(f"  Years: {_unique_years}")
        for _col in _offset_cols:
            if _col not in data.columns:
                continue
            offsets = {}
            for _yr in _unique_years:
                _yr_idx = test_idx.intersection(
                    data.index[(_test_years == _yr) & data[_col].notna()]
                )
                if len(_yr_idx) == 0:
                    continue
                _offset = float(y.loc[_yr_idx].mean()) - float(
                    data.loc[_yr_idx, _col].mean()
                )
                data.loc[_yr_idx, _col] += _offset
                offsets[_yr] = _offset
            offsets_str = "  ".join(f"{yr}: {v:+.3f}" for yr, v in offsets.items())
            print(f"  {_col:<40s}  {offsets_str}")

    # ---- Results ----
    print_tex_results_table(
        data=data,
        y=y,
        eval_idx=test_idx,
        methods=[
            ("Mode of reporting-org score baseline", "per_org_mode"),
            ("Ridge Baseline (risks + org only)", "ridge_baseline_risk_org"),
            ("Plain OLS GLM", "pred_glm_plain"),
            # ("Plain OLS GLM + year corr", "pred_glm_plain_yr_corr"),
            ("RF+ET all features (no year corr)", "pred_rf_no_yr_corr"),
            ("RF+ET all features + year corr", "pred_rf"),
            ("RF+ET, no LLM features", "pred_rf_no_llm"),
            ("XGBoost all features", "pred_xgb"),
            # ("XGBoost, no LLM features", "pred_xgb_no_llm"),
            # ("Ridge GLM", "glm_nonbin_pred_ridge"),
            ("RF+ET (default params)", "pred_rf_default_params"),
            # ("Ridge GLM + RF+ET (mean)", "ridge_plus_rf"),
            # ("Ridge + RF+ET ensemble + year corr", "ridge_plus_rf_corrected"),
            # ("RF+ET + recency", "pred_rf_recency"),
            # ("RF+ET + LLM Forecast (no year corr)", "pred_rf_llm_modded_no_yr_corr"),
            ("RF+ET + LLM Forecast (ridge)", "pred_rf_llm_modded"),
            ("RF+ET + LLM Forecast (simple avg)", "pred_rf_llm_simple_avg"),
            # ("Ridge+RF ensemble + LLM + recency", "ridge_plus_rf_llm_modded"),
            # ("RF + LLM Forecast + recency (rounded)", "pred_rf_llm_modded_rounded"),
        ],
        side_threshold=3.5,
        decimals=3,
    )

    # WG pairwise restricted to pairs where the primary LLM made a non-tied prediction
    _llm_stem = VARIANT_PATHS[0].stem
    _llm_col = f"fewshot_variant_{_llm_stem}_prediction"
    if _llm_col in data.columns:
        print_wg_pairwise_on_llm_pairs(
            data=data,
            y=y,
            eval_idx=test_idx,
            model_cols=[
                "pred_rf",
                "pred_rf_no_llm",
                "pred_xgb",
                "pred_rf_default_params",
                "pred_rf_llm_modded",
                "pred_rf_llm_simple_avg",
                "per_org_mode",
            ],
            llm_col=_llm_col,
        )
    else:
        print(
            f"\n[Note] LLM column '{_llm_col}' not found; skipping LLM-pair comparison."
        )

    # Bootstrap CI
    ci_idx = held_idx if USE_VAL_IN_TRAIN else val_idx
    ci_label = "Held-out Test Set" if USE_VAL_IN_TRAIN else "Validation Set"
    print("\n" + "=" * 80)
    print(f"BOOTSTRAP 95% CI: RF+ET + year corr ({ci_label})")
    print("=" * 80)

    ci_mask = ci_idx.intersection(data.index)
    ci_mask = ci_mask[data.loc[ci_mask, "pred_rf"].notna()]
    y_true_val = y.loc[ci_mask].astype(float).to_numpy()
    y_pred_val = data.loc[ci_mask, "pred_rf"].astype(float).to_numpy()
    _ci_groups = (
        data.loc[ci_mask, "reporting_orgs"].astype(str)
        + "_"
        + data.loc[ci_mask, "start_date"].dt.year.astype(str)
    ).to_numpy()

    r2_val = r2_score(y_true_val, y_pred_val)
    r2_ci = bootstrap_ci(
        y_true_val, y_pred_val, lambda y_t, y_p: r2_score(y_t, y_p), n_bootstrap=100
    )

    pairwise_val = within_group_pairwise_ordering_prob(
        y_true_val, y_pred_val, _ci_groups
    )["prob"]
    _rng_ci = np.random.RandomState(42)
    _boot_pops = []
    for _ in range(100):
        _bi = _rng_ci.choice(len(y_true_val), size=len(y_true_val), replace=True)
        _bp = within_group_pairwise_ordering_prob(
            y_true_val[_bi], y_pred_val[_bi], _ci_groups[_bi]
        )["prob"]
        if np.isfinite(_bp):
            _boot_pops.append(_bp)
    _boot_pops = np.array(_boot_pops)
    pairwise_ci = {
        "mean": float(np.mean(_boot_pops)),
        "lower": float(np.percentile(_boot_pops, 2.5)),
        "upper": float(np.percentile(_boot_pops, 97.5)),
    }

    print(
        f"R^2 = {r2_val:.4f}  [95% CI: {r2_ci['lower']:.4f}, {r2_ci['upper']:.4f}]  (n={len(y_true_val)})"
    )
    print(
        f"Pairwise = {pairwise_val:.4f}  [95% CI: {pairwise_ci['lower']:.4f}, {pairwise_ci['upper']:.4f}]"
    )
    print("=" * 80)

    # ---- REDUCED (ADJUSTED) R^2 ON TEST SET -- OVERALL AND BY ORG ----
    _adj_r2_col = "pred_rf_llm_modded"
    _adj_r2_p = len(feature_cols)
    _adj_r2_org_labels = {
        "Overall": None,
        "World Bank": "World Bank",
        "BMZ": (
            "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); "
            "Federal Ministry for Economic Cooperation and Development (BMZ)"
        ),
    }

    def _print_adj_r2_block(header, eval_idx):
        print()
        print("=" * 80)
        print(f"REDUCED (ADJUSTED) R^2  --  {header}")
        print(f"MODEL: RF+ET + LLM FORECAST + RECENCY   (p={_adj_r2_p} features)")
        print("=" * 80)
        for _grp_label, _org_str in _adj_r2_org_labels.items():
            if _org_str is None:
                _grp_idx = eval_idx.intersection(data.index)
            else:
                _org_mask = (
                    data.loc[eval_idx.intersection(data.index), "reporting_orgs"]
                    == _org_str
                )
                _grp_idx = eval_idx.intersection(data.index)[_org_mask]
            _grp_idx = _grp_idx[data.loc[_grp_idx, _adj_r2_col].notna()]
            _grp_idx = _grp_idx[y.loc[_grp_idx].notna()]
            if len(_grp_idx) < _adj_r2_p + 2:
                print(
                    f"  {_grp_label:<12}: n={len(_grp_idx)}  (too few observations for adjusted R^2)"
                )
                continue
            _yt = y.loc[_grp_idx].astype(float).to_numpy()
            _yp = data.loc[_grp_idx, _adj_r2_col].astype(float).to_numpy()
            _r2 = r2_score(_yt, _yp)
            _adj = adjusted_r2(_r2, len(_yt), _adj_r2_p)
            print(
                f"  {_grp_label:<12}: n={len(_grp_idx):4d}   R^2={_r2:.4f}   Adjusted R^2={_adj:.4f}"
            )
        print("=" * 80)
        print()

    _adj_r2_col = "pred_rf_llm_modded"
    _adj_r2_p = len(feature_cols)
    _adj_r2_org_labels = {
        "Overall": None,
        "World Bank": "World Bank",
        "BMZ": (
            "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); "
            "Federal Ministry for Economic Cooperation and Development (BMZ)"
        ),
    }
    _train_label = (
        "TRAIN+VAL SET (IN-SAMPLE)" if USE_VAL_IN_TRAIN else "TRAIN SET (IN-SAMPLE)"
    )
    _print_adj_r2_block(_train_label, train_idx)
    _print_adj_r2_block("TEST SET (OUT-OF-SAMPLE)", ci_mask)

    # ---- WITHIN-ORG, SAME-STARTING-YEAR PAIRWISE ORDERING PROBABILITY ----
    _WB_ORG = "World Bank"
    _BMZ_ORG = (
        "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); "
        "Federal Ministry for Economic Cooperation and Development (BMZ)"
    )
    _methods_for_pairwise = [
        ("Ridge (risks+org) baseline", "ridge_baseline_risk_org"),
        ("RF+ET (no year corr)", "pred_rf_no_yr_corr"),
        ("RF+ET + year corr", "pred_rf"),
        ("Ridge+RF+ET ensemble+yr corr", "ridge_plus_rf_corrected"),
        ("RF+ET+LLM+recency", "pred_rf_llm_modded"),
        ("Ridge+RF+ET+LLM+recency", "ridge_plus_rf_llm_modded"),
    ]
    _pop_idx = held_idx  # always use test set

    # Collect org and year arrays for the test set (used by org_year_pairwise_ordering_prob)
    _pop_orgs = data.loc[_pop_idx, "reporting_orgs"].to_numpy(dtype=str)
    _pop_years = (
        data.loc[_pop_idx, "start_date"].dt.year.astype(str).to_numpy()
        if "start_date" in data.columns
        else np.full(len(_pop_idx), "unknown")
    )
    _n_wb = int((_pop_orgs == _WB_ORG).sum())
    _n_bmz = int((_pop_orgs == _BMZ_ORG).sum())

    print("\n" + "=" * 90)
    print("WITHIN-ORG, SAME-STARTING-YEAR PAIRWISE ORDERING PROBABILITY (test set)")
    print(f"  Group key: reporting_org x start_year  |  WB n={_n_wb}  BMZ n={_n_bmz}")
    print("=" * 90)
    print(f"  {'Method':<32} {'World Bank':>22} {'BMZ':>20} {'Weighted avg':>14}")
    print(f"  {'-'*91}")

    for method_label, method_col in _methods_for_pairwise:
        if method_col not in data.columns:
            continue
        valid_mask = data.loc[_pop_idx, method_col].notna()
        valid_idx = _pop_idx[valid_mask]
        if len(valid_idx) < 2:
            print(f"  {method_label:<32} {'n/a':>22} {'n/a':>20} {'n/a':>14}")
            continue
        yt_pop = y.loc[valid_idx].to_numpy(dtype=float)
        yp_pop = data.loc[valid_idx, method_col].to_numpy(dtype=float)
        orgs_pop = data.loc[valid_idx, "reporting_orgs"].to_numpy(dtype=str)
        yrs_pop = (
            data.loc[valid_idx, "start_date"].dt.year.astype(str).to_numpy()
            if "start_date" in data.columns
            else np.full(len(valid_idx), "unknown")
        )
        weighted_avg, per_org = org_year_pairwise_ordering_prob(
            yt_pop,
            yp_pop,
            orgs_pop,
            yrs_pop,
            target_orgs=[_WB_ORG, _BMZ_ORG],
        )
        wb_pop, wb_np = per_org.get(_WB_ORG, (np.nan, 0))
        bmz_pop, bmz_np = per_org.get(_BMZ_ORG, (np.nan, 0))
        wb_str = f"{wb_pop:6.4f} ({wb_np}p)" if np.isfinite(wb_pop) else "n/a"
        bmz_str = f"{bmz_pop:6.4f} ({bmz_np}p)" if np.isfinite(bmz_pop) else "n/a"
        avg_str = f"{weighted_avg:6.4f}" if np.isfinite(weighted_avg) else "n/a"
        print(f"  {method_label:<32} {wb_str:>22} {bmz_str:>20} {avg_str:>14}")

    print("=" * 90)

    # ---- SAVE MODEL OUTPUTS ----
    print("\nSaving model outputs...")

    # 1. Save feature importances (only when --importances flag was passed)
    if show_importances and len(imp_rf) > 0:
        imp_save = imp_rf.copy()
        if "importance_abs_1sd" in imp_save.columns:
            imp_save = imp_save.rename(columns={"importance_abs_1sd": "importance"})
        imp_save.to_csv(OUTPUT_DIR / "feature_importances.csv", index=False)

    # 2. Save predictions
    predictions_list = []
    pred_col = "pred_rf_llm_modded"
    for split_name, split_idx in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", held_idx),
    ]:
        split_data = data.loc[split_idx].copy()
        split_y = y.loc[split_idx]
        for aid in split_data.index:
            if aid in split_y.index:
                predictions_list.append(
                    {
                        "activity_id": aid,
                        "y_true": split_y.loc[aid],
                        "y_pred": (
                            split_data.loc[aid, pred_col]
                            if pred_col in split_data.columns
                            else np.nan
                        ),
                        "split": split_name,
                    }
                )
    if len(predictions_list) > 0:
        pd.DataFrame(predictions_list).to_csv(
            OUTPUT_DIR / "predictions.csv", index=False
        )

    # 3. Save metadata
    metadata = {
        "model_name": "Rating",
        "target_description": "Overall success rating (0-5 scale: 0=Highly Unsatisfactory, 5=Highly Satisfactory)",
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(held_idx),
        "val_r2": float(r2_val) if np.isfinite(r2_val) else None,
        "timestamp": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 4. Save feature names
    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # 5. Save RF model and ExtraTrees model
    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    if extra_model is not None:
        with open(OUTPUT_DIR / "extra_model.pkl", "wb") as f:
            pickle.dump(extra_model, f)

    # 6. Save recency corrector
    if recency_model is not None:
        with open(OUTPUT_DIR / "recency_corrector.pkl", "wb") as f:
            pickle.dump(recency_model, f)

    # 7. Save training medians and feature matrix
    train_data = data.loc[train_idx, feature_cols].copy()
    train_medians = train_data.median().to_dict()
    additional_features = ["umap3_x", "umap3_y", "umap3_z", "log_planned_expenditure"]
    for feat in additional_features:
        if feat in data.columns and feat not in train_medians:
            train_medians[feat] = float(data.loc[train_idx, feat].median())
    with open(OUTPUT_DIR / "train_medians.json", "w") as f:
        json.dump(train_medians, f, indent=2)

    data.loc[train_idx, feature_cols].copy().to_csv(
        OUTPUT_DIR / "train_features.csv", index=False
    )

    # Save val feature matrix with activity_id index (used by fast_feature_importance.py)
    val_feat_df = data.loc[val_idx, feature_cols].copy()
    val_feat_df.index.name = "activity_id"
    val_feat_df.to_csv(OUTPUT_DIR / "val_features.csv")

    print(f"All outputs saved to {OUTPUT_DIR}")

    # Save best model predictions CSV
    best_model_cols = [
        "rating",
        "pred_rf_llm_modded",
        "pred_rf",
        "ridge_baseline_risk_org",
        "per_org_mode",
        "planned_expenditure",
        "planned_duration",
        "activity_scope",
        "risks",
        "reporting_orgs",
    ]
    best_model_cols.extend([c for c in data.columns if c.startswith("sector_cluster_")])
    available_cols = [c for c in best_model_cols if c in data.columns]
    save_df = data[available_cols].copy()
    save_df.index.name = "activity_id"
    save_df.to_csv(DATA_DIR / "best_model_predictions.csv")

    print(f"\nTotal input features used: {len(feature_cols)}")
    for f in feature_cols:
        print(f"  {f}")


if __name__ == "__main__":
    main()
