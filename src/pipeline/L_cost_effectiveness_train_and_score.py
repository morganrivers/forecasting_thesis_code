"""
Trains and evaluates cost-effectiveness prediction models using RF/ET ensembles,
with SHAP feature importance analysis and out-of-sample scoring.
"""

import pickle
import os
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Set
from collections import Counter
import re
import pprint
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import glob
import shap
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from collections import defaultdict

# scipy.stats.spearmanr now wrapped in scoring_metrics.spearman_correlation

import sys

ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS = [
    "out_raw_benefit_cost_ratios__ratio",
    # 'out_raw_yield_increases_percent__percent',
    # 'out_raw_yield_increases_t_per_ha__t_per_hectare',
    "out_raw_economic_rate_of_return__percent",
    "out_raw_financial_rate_of_return__percent",
    # 'out_dpu_area_protected__hectares',
    # 'out_dpu_area_reforested__hectares',
    # 'out_dpu_area_under_management__hectares',
    # 'out_dpu_beneficiaries__people',
    "out_dpu_co2_emission_reductions__tonnes_co2e",
    "out_dpu_co2_emission_reductions__tonnes_co2e_per_year",
    "out_dpu_generation_capacity__gwh",
    "out_dpu_generation_capacity__mw",
    # 'out_dpu_pollution_load_removed__tonnes_per_year',
    # 'out_dpu_stoves__count_stoves',
    # 'out_dpu_trees_planted__count',
    "out_dpu_water_connections__connections",
    # 'out_raw_area_protected__hectares',
    # 'out_raw_area_reforested__hectares',
    # 'out_raw_area_under_management__hectares',
    # # 'out_raw_beneficiaries__people',
    # 'out_raw_co2_emission_reductions__tonnes_co2e',
    # 'out_raw_co2_emission_reductions__tonnes_co2e_per_year',
    # 'out_raw_generation_capacity__gwh',
    # 'out_raw_generation_capacity__mw',
    # 'out_raw_pollution_load_removed__tonnes_per_year',
    # 'out_raw_stoves__count_stoves',
    # 'out_raw_trees_planted__count',
    # 'out_raw_water_connections__connections',
    # "out_rating"
]


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UTILS_DIR = REPO_ROOT / "src" / "utils"
DATA_DIR = REPO_ROOT / "data"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from ml_models import (
    run_random_forest_median_impute_noclip,
    run_ridge_glm_median_impute_noclip,
    one_sd_shift_importance,
    bootstrap_ci,
    apply_start_year_trend_correction,
)
from scoring_metrics import (
    rmse,
    mae,
    r2 as r2_metric,
    true_hit_accuracy,
    side_accuracy,
    spearman_correlation,
    pairwise_ordering_prob_excl_ties as pairwise_ordering_prob,
    within_group_pairwise_ordering_prob,
)

from data_similar_activities import find_similar_activities_semantic
from feature_engineering import (
    get_success_measure_from_rating_value_wrapped,
    load_grades,
    load_is_completed,
    load_ratings,
    load_activity_scope,
    load_gdp_percap,
    load_implementing_org_type,
    load_world_bank_indicators,
    add_similarity_features,
    pick_start_date,
    parse_last_line_label_after_forecast,
    add_dates_to_dataframe,
    restrict_to_reporting_orgs_exact,
    load_targets_context_maps_features,
    add_enhanced_uncertainty_features,
    data_sector_clusters,
)
from data_loan_disbursement import load_loan_or_disbursement
from data_currency_conversion import return_misc_disbursement_or_planned_disbursement

MODEL_TO_USE = "logit_and_ordinal"

NUM_ORGS_KEEP = 4
from split_constants import (
    LATEST_TRAIN_POINT,
    LATEST_VALIDATION_POINT,
    TOO_LATE_CUTOFF,
    split_latest_by_date_with_cutoff as _split_canonical,
)

USE_SPLIT = True
USE_CACHE = False
KEEP_REPORTING_ORGS = [
    "UK - Foreign, Commonwealth Development Office (FCDO)",
    "Asian Development Bank",
    "World Bank",
    "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); Federal Ministry for Economic Cooperation and Development (BMZ)",
]
DROP_INCOMPLETE = True
USE_VAL_IN_TRAIN = True
ADD_INCOMPLETE_TO_TEST = False
SKIP_RATINGS_WHEN_REPORTING = True
DISABLE_ALL_BUT_ZAGG = False  # Skip individual outcome models, only run zagg
DONT_MAKE_ANY_SHAP_ANALYSIS_OR_PLOTS = True  # Skip SHAP and all plotting
ADD_INTERACTION_FEATURES = False
ADD_GROUP_INTERACTION_FEATURES = False

SUBTRACT_MEAN_IN_Z_SCORE = True

# Option A: Replace outcome with ratio when both exist (at row level in outcomes_df)
USE_OUTCOME_TARGET_RATS = False

# Option B: Add ratio as additional outcome in zagg (alongside other outcomes)
INCLUDE_RATIO_IN_ZAGG = False

# Option C: Only use ratio for specific distributions (e.g., beneficiaries)
USE_RATIO_FOR_SPECIFIC_DISTS_ONLY = False
RATIO_ONLY_DISTRIBUTIONS = [
    "beneficiaries",
    "benefit_cost_ratios",
    "yield_increases_percent",
    "yield_increases_t_per_ha",
    "economic_rate_of_return",
    "financial_rate_of_return",
    # Add other distributions here if needed (e.g., "people_with_access_to_water")
]

# Only applies when INCLUDE_RATIO_IN_ZAGG=True: removes ratio from val/test sets
REMOVE_RATIO_FROM_VAL_AND_TEST = False

# Minimum number of rows per outcome group for zagg z-score calculation
# Each outcome group must have at least this many rows in train AND test to be included
MIN_TRAIN_ROWS_PER_GROUP = 1  # Min rows in training set per outcome group
MIN_TEST_ROWS_PER_GROUP = 1  # Min rows in test set per outcome group

USE_RATING_AS_FEATURE_AND_DONT_PREDICT_IT = False
USE_SAME_FEATURES_AS_OVERALL_RATING = (
    True  # if True, aligns features with A_overall_rating_fit_and_evaluate.py
)
GET_FEATURE_IMPORTANCES = True

RESTRICT_TO_REPORTING_ORGS = True

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ALL_GRADES = "../../data/*_grades.jsonl"
OUT_MISC = Path("../../data/outputs_misc.jsonl")
INFO_FOR_ACTIVITY_FORECASTING = (
    "../../data/info_for_activity_forecasting_old_transaction_types.csv"
)
MERGED_OVERALL_RATINGS = "../../data/merged_overall_ratings.jsonl"
TARGETS_CONTEXT_MAPS = Path("../../data/outputs_targets_context_maps.jsonl")
FINANCE_SECTORS = (
    "../../data/outputs_finance_sectors_disbursements_baseline_gemini2p5flash.jsonl"
)
ACTIVITY_OUTCOMES_PATH = Path("../../data/activity_outcomes.csv")
CACHE_FILE = DATA_DIR / "prediction_outcome_cached_input_data.pkl"
EXCLUDE_DPU_DISTS = {
    "benefit_cost_ratios",
    "economic_rate_of_return",
    "financial_rate_of_return",
}


def is_yield_dist(dist: str) -> bool:
    d = str(dist).lower()
    return ("yield_increase" in d) or ("yield_increases" in d) or ("yield" in d)


def zagg_metrics_per_outcome(
    long_df: pd.DataFrame,
    *,
    outcome_cols: list[str],
    meta_by_col: dict,
    test_aids: set[str],
    group_col: str = "which_group",
    y_col: str = "y_z",
    pred_col: str = "pred_z",
    min_rows: int = 2,
):
    """
    Evaluate the single long-form z model (pred_z) on the subset of test rows
    corresponding to each outcome column (i.e., each which_group).
    Returns a df keyed by outcome_col with z-model R^2/RMSE/MAE and n.
    """
    rows = []
    for oc in outcome_cols:
        meta = meta_by_col.get(oc, {})
        dist = meta.get("which_distribution")
        units = meta.get("which_units")

        # must match build_long_outcome_rows() group labeling
        grp = f"{dist}__{units}" if (dist is not None and units is not None) else oc

        m = (long_df["activity_id"].isin(test_aids)) & (long_df[group_col] == grp)
        yt = pd.to_numeric(long_df.loc[m, y_col], errors="coerce").astype(float)
        yp = pd.to_numeric(long_df.loc[m, pred_col], errors="coerce").astype(float)
        mm = np.isfinite(yt.values) & np.isfinite(yp.values)
        n = int(mm.sum())
        if n < min_rows:
            continue

        r2 = float(r2_score(yt.values[mm], yp.values[mm]))
        rmse = float(np.sqrt(np.mean((yp.values[mm] - yt.values[mm]) ** 2)))
        mae = float(np.mean(np.abs(yp.values[mm] - yt.values[mm])))

        rows.append(
            {
                "outcome_col": oc,
                "zmodel_group": grp,
                "zmodel_r2": r2,
                "zmodel_rmse": rmse,
                "zmodel_mae": mae,
                "zmodel_n_rows": n,
            }
        )

    return pd.DataFrame(rows)


def build_long_outcome_rows(
    data_wide: pd.DataFrame,
    *,
    outcome_cols: List[str],
    feature_cols: List[str],
    meta_by_col: Dict[str, Dict[str, Any]] = None,
    group_col_name: str = "which_group",
):
    """
    Returns long_df indexed by a unique row id, with columns:
      - activity_id
      - which_group  (string)
      - y_raw        (float outcome)
      - feature cols copied from data_wide
    """
    meta_by_col = meta_by_col or {}

    rows = []
    for oc in outcome_cols:
        if oc not in data_wide.columns:
            continue
        s = pd.to_numeric(data_wide[oc], errors="coerce")
        m = s.notna()
        if not m.any():
            continue

        # group label: prefer meta if available, otherwise column name
        meta = meta_by_col.get(oc, {})
        dist = meta.get("which_distribution")
        units = meta.get("which_units")
        if dist is not None and units is not None:
            grp = f"{dist}__{units}"
        else:
            grp = oc

        df_part = data_wide.loc[m, feature_cols].copy()
        df_part["activity_id"] = df_part.index.astype(str)
        df_part[group_col_name] = grp
        df_part["y_raw"] = s.loc[m].astype(float).values
        rows.append(df_part)

    long_df = pd.concat(rows, axis=0, ignore_index=True)

    # unique row index (so duplicates of activity_id are fine)
    long_df["row_id"] = (
        long_df["activity_id"].astype(str)
        + "__"
        + long_df[group_col_name].map(_safe_name).astype(str)
        + "__"
        + np.arange(len(long_df)).astype(str)
    )
    long_df = long_df.set_index("row_id", drop=True)

    return long_df


def split_activity_ids_by_date_with_cutoff(base_df: pd.DataFrame, date_col: str):
    """
    Uses split_latest_by_date_with_cutoff, but returns activity_id sets.
    base_df must be indexed by activity_id.
    """
    tr, va, te = split_latest_by_date_with_cutoff(base_df, date_col=date_col)
    return set(map(str, tr)), set(map(str, va)), set(map(str, te))


def filter_groups_min_counts(
    long_df: pd.DataFrame,
    *,
    train_aids: Set[str],
    test_aids: Set[str],
    group_col: str = "which_group",
    min_train: int = 10,
    min_test: int = 10,
):
    """
    Keep only groups that have >=min_train labeled rows in TRAIN and >=min_test labeled rows in TEST.
    """
    is_train = long_df["activity_id"].isin(train_aids)
    is_test = long_df["activity_id"].isin(test_aids)

    ct_train = long_df.loc[is_train].groupby(group_col).size()
    ct_test = long_df.loc[is_test].groupby(group_col).size()

    good = ct_train.index.intersection(ct_test.index)
    good = [g for g in good if (ct_train[g] >= min_train and ct_test[g] >= min_test)]

    out = long_df[long_df[group_col].isin(good)].copy()

    print("\n[group filter] kept groups:", len(good))
    print("[group filter] rows before:", len(long_df), "after:", len(out))
    if len(good):
        print("[group filter] min train rows among kept:", int(ct_train[good].min()))
        print("[group filter] min test  rows among kept:", int(ct_test[good].min()))

    return out, good


def add_groupwise_z(
    long_df: pd.DataFrame,
    *,
    train_aids: Set[str],
    group_col: str = "which_group",
    y_col: str = "y_raw",
    z_col: str = "y_z",
    median_col: str = "y_median",
    min_sd: float = 1e-12,
):
    """
    Compute z = (y - mu_g)/sd_g using TRAIN-only stats per group.
    Also compute median_g for use as prediction target.
    Adds columns mu_g, sd_g, median_g, y_z (NaN if group missing stats).
    """
    is_train = long_df["activity_id"].isin(train_aids)
    g = long_df.loc[is_train].groupby(group_col)[y_col]
    mu = g.mean()
    median = g.median()
    sd = g.std(ddof=0)

    # map onto all rows
    long_df["mu_g"] = long_df[group_col].map(mu).astype(float)
    long_df["median_g"] = long_df[group_col].map(median).astype(float)
    long_df["sd_g"] = long_df[group_col].map(sd).astype(float)

    bad_sd = (~np.isfinite(long_df["sd_g"])) | (long_df["sd_g"] < min_sd)
    long_df.loc[bad_sd, "sd_g"] = np.nan

    # Z-score using mean and sd
    long_df[z_col] = (long_df[y_col].astype(float) - long_df["mu_g"]) / long_df["sd_g"]

    # Median-based target
    long_df[median_col] = long_df[y_col].astype(float) - long_df["median_g"]

    return long_df


def add_group_dummies(
    long_df: pd.DataFrame, group_col: str = "which_group", prefix: str = "grp__"
):
    d = pd.get_dummies(long_df[group_col].astype(str), prefix=prefix, dtype=float)
    long_df = pd.concat([long_df, d], axis=1)
    return long_df, list(d.columns)


def load_activity_outcomes_df(
    path=ACTIVITY_OUTCOMES_PATH,
    *,
    dedupe=True,
    drop_missing_keys=True,
    require_value_cols=("outcome_norm",),  # used only for universe filtering
):
    path = Path(path)
    print("\n=== LOADING ACTIVITY OUTCOMES ===")
    print("Path:", path.resolve())

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported outcomes file type: {path.suffix}")

    needed = ["activity_id", "which_distribution", "which_units"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Outcomes file missing required column: {c}")

    df["activity_id"] = df["activity_id"].astype(str).str.strip()
    df["which_distribution"] = df["which_distribution"].astype(str)
    df["which_units"] = df["which_units"].astype(str)

    # coerce all numeric columns we might use
    num_cols = [
        "outcome_norm",
        "baseline_norm",
        "target_norm",
        "outcome_norm_log10",
        "baseline_norm_log10",
        "target_norm_log10",
        "actual_total_expenditure",
        "outcome_norm_dollars_per_unit",
        "baseline_norm_dollars_per_unit",
        "target_norm_dollars_per_unit",
        "outcome_norm_dollars_per_unit_log10",
        "baseline_norm_dollars_per_unit_log10",
        "target_norm_dollars_per_unit_log10",
        "expenditure_alloc",
        "outcome_norm_dollars_per_unit_split",
        "outcome_norm_dollars_per_unit_split_log10",
        "baseline_norm_dollars_per_unit_split",
        "baseline_norm_dollars_per_unit_split_log10",
        "target_norm_dollars_per_unit_split",
        "target_norm_dollars_per_unit_split_log10",
        "outcome_over_target",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if drop_missing_keys:
        before = len(df)
        df = df.dropna(subset=needed)
        print("Drop: missing key cols:", before - len(df))

    # require at least one of the requested value cols to be present
    have_cols = [c for c in require_value_cols if c in df.columns]
    if have_cols:
        before = len(df)
        df = df[df[have_cols].notna().any(axis=1)].copy()
        print(f"Drop: all {have_cols} are NaN:", before - len(df))

    # dedupe key
    key = ["activity_id", "which_distribution", "which_units"]
    dup_n = int(df.duplicated(subset=key).sum())
    print("Duplicate key rows:", dup_n)

    if dedupe and dup_n > 0:
        agg = {}
        for c in num_cols:
            if c in df.columns and c not in key:
                agg[c] = "mean"
        before = len(df)
        df = df.groupby(key, as_index=False).agg(agg)
        print("Deduped rows:", before, "->", len(df))

    print("Rows:", len(df), "Unique activity_id:", df["activity_id"].nunique())
    return df


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:120]


def top1_abs_1sd_shift_importance(
    *,
    model,
    data: pd.DataFrame,
    feature_cols: list[str],
    train_idx,
    n_eval: int = 200,
    random_state: int = 0,
):
    """
    Cheap model-based importance:
      For each feature j, compute mean |f(x + sd_j e_j) - f(x)| over a small sample of rows.
      sd_j comes from TRAIN (median-imputed).
    Returns dict with:
      - top1_feat
      - top1_abs_1sd  (in target units)
    """
    tr = pd.Index(train_idx)
    Xtr = data.loc[tr, feature_cols].astype(float)
    med = Xtr.median(numeric_only=True)
    sd = Xtr.fillna(med).std(numeric_only=True)

    # sample evaluation rows (use TRAIN to be stable and cheap)
    X = data.loc[tr, feature_cols].astype(float).fillna(med)
    if len(X) == 0:
        return {"top1_feat": None, "top1_abs_1sd": np.nan}

    Xs = X.sample(min(int(n_eval), len(X)), random_state=random_state)

    base = model.predict(Xs)
    base = np.asarray(base, dtype=float).ravel()

    scores = {}
    for j in feature_cols:
        s = float(sd.get(j, np.nan))
        if not np.isfinite(s) or s <= 0:
            continue
        Xp = Xs.copy()
        Xp[j] = Xp[j] + s
        yp = np.asarray(model.predict(Xp), dtype=float).ravel()
        scores[j] = float(np.mean(np.abs(yp - base)))

    if not scores:
        return {"top1_feat": None, "top1_abs_1sd": np.nan}

    top_feat = max(scores, key=scores.get)
    return {"top1_feat": top_feat, "top1_abs_1sd": float(scores[top_feat])}


# ---------------------------------------------------------------------------
# PREDICTION PARSERS + GENERIC JSONL LOADER
# ---------------------------------------------------------------------------
def outcomes_to_wide_two_variants(outcomes_df: pd.DataFrame):
    tmp = outcomes_df.copy()
    tmp["activity_id"] = tmp["activity_id"].astype(str).str.strip()
    tmp["which_distribution"] = tmp["which_distribution"].astype(str)
    tmp["which_units"] = tmp["which_units"].astype(str)

    def _safe(s):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))[:120]

    # ---- RAW ----
    if "outcome_norm" not in tmp.columns or "outcome_norm_log10" not in tmp.columns:
        raise ValueError(
            "Need outcome_norm and outcome_norm_log10 in outcomes_df (from the CSV writer)."
        )

    dist_l = tmp["which_distribution"].astype(str).str.lower()
    is_yield = dist_l.map(is_yield_dist)
    is_ror = dist_l.isin({"economic_rate_of_return", "financial_rate_of_return"})

    raw = tmp.copy()
    # yield + RoR use linear; everything else uses log10
    if USE_OUTCOME_TARGET_RATS:
        # Use ratio when both outcome and target exist, otherwise use normal logic
        raw["_val"] = np.where(
            raw["outcome_over_target"].notna(),
            raw["outcome_over_target"],
            np.where(is_yield | is_ror, raw["outcome_norm"], raw["outcome_norm_log10"]),
        )
        n_ratio = raw["outcome_over_target"].notna().sum()
        n_total_before = len(raw)
        print(
            f"\n[USE_OUTCOME_TARGET_RATS] RAW: Using ratio for {n_ratio}/{n_total_before} rows ({100*n_ratio/n_total_before:.1f}%)"
        )
    elif USE_RATIO_FOR_SPECIFIC_DISTS_ONLY:
        # Only use ratio for specific distributions
        dist_lower = raw["which_distribution"].astype(str).str.lower()
        use_ratio = raw["outcome_over_target"].notna() & dist_lower.isin(
            [d.lower() for d in RATIO_ONLY_DISTRIBUTIONS]
        )
        raw["_val"] = np.where(
            use_ratio,
            raw["outcome_over_target"],
            np.where(is_yield | is_ror, raw["outcome_norm"], raw["outcome_norm_log10"]),
        )
        n_ratio = use_ratio.sum()
        n_total_before = len(raw)
        print(
            f"\n[USE_RATIO_FOR_SPECIFIC_DISTS_ONLY] RAW: Using ratio for {n_ratio}/{n_total_before} rows ({100*n_ratio/n_total_before:.1f}%)"
        )
        print(f"  Distributions using ratio: {RATIO_ONLY_DISTRIBUTIONS}")
    else:
        raw["_val"] = np.where(
            is_yield | is_ror, raw["outcome_norm"], raw["outcome_norm_log10"]
        )
    raw = raw[raw["_val"].notna()].copy()
    raw["colname"] = raw.apply(
        lambda r: f"out_raw_{_safe(r['which_distribution'])}__{_safe(r['which_units'])}",
        axis=1,
    )

    meta_raw = (
        raw[["colname", "which_distribution", "which_units"]]
        .drop_duplicates()
        .set_index("colname")[["which_distribution", "which_units"]]
        .to_dict(orient="index")
    )

    wide_raw = raw.pivot_table(
        index="activity_id", columns="colname", values="_val", aggfunc="mean"
    ).sort_index()
    raw_cols = list(wide_raw.columns.astype(str))

    print("\n[outcomes] RAW wide:", wide_raw.shape, "cols:", len(raw_cols))

    if USE_SPLIT:

        dpu_needed = ["outcome_norm_dollars_per_unit_split_log10"]
        for c in dpu_needed:
            if c not in tmp.columns:
                raise ValueError(f"Need {c} in outcomes_df (from the CSV writer).")

        dpu = tmp.copy()
        dpu_dist_l = dpu["which_distribution"].astype(str).str.lower()
        dpu_is_yield = dpu_dist_l.map(is_yield_dist)
        dpu_is_excluded = dpu_dist_l.isin(EXCLUDE_DPU_DISTS)

        # exclude yield + excluded dists
        dpu = dpu[~dpu_is_yield & ~dpu_is_excluded].copy()

        # USE SPLIT
        if USE_OUTCOME_TARGET_RATS:
            # Use ratio when both outcome and target exist, otherwise use per-dollar outcome
            dpu["_val"] = np.where(
                dpu["outcome_over_target"].notna(),
                dpu["outcome_over_target"],
                dpu["outcome_norm_dollars_per_unit_split_log10"],
            )
            n_ratio = dpu["outcome_over_target"].notna().sum()
            n_total_before = len(dpu)
            print(
                f"[USE_OUTCOME_TARGET_RATS] DPU (SPLIT): Using ratio for {n_ratio}/{n_total_before} rows ({100*n_ratio/n_total_before:.1f}%)"
            )
        elif USE_RATIO_FOR_SPECIFIC_DISTS_ONLY:
            # Only use ratio for specific distributions
            dist_lower = dpu["which_distribution"].astype(str).str.lower()
            use_ratio = dpu["outcome_over_target"].notna() & dist_lower.isin(
                [d.lower() for d in RATIO_ONLY_DISTRIBUTIONS]
            )
            dpu["_val"] = np.where(
                use_ratio,
                dpu["outcome_over_target"],
                dpu["outcome_norm_dollars_per_unit_split_log10"],
            )
            n_ratio = use_ratio.sum()
            n_total_before = len(dpu)
            print(
                f"[USE_RATIO_FOR_SPECIFIC_DISTS_ONLY] DPU (SPLIT): Using ratio for {n_ratio}/{n_total_before} rows ({100*n_ratio/n_total_before:.1f}%)"
            )
        else:
            dpu["_val"] = dpu["outcome_norm_dollars_per_unit_split_log10"]
        dpu = dpu[dpu["_val"].notna()].copy()

        dpu["colname"] = dpu.apply(
            lambda r: f"out_dpu_{_safe(r['which_distribution'])}__{_safe(r['which_units'])}",
            axis=1,
        )

        meta_dpu = (
            dpu[["colname", "which_distribution", "which_units"]]
            .drop_duplicates()
            .set_index("colname")[["which_distribution", "which_units"]]
            .to_dict(orient="index")
        )

        wide_dpu = dpu.pivot_table(
            index="activity_id", columns="colname", values="_val", aggfunc="mean"
        ).sort_index()
        dpu_cols = list(wide_dpu.columns.astype(str))

    else:

        # ---- DPU ----
        dpu_needed = ["outcome_norm_dollars_per_unit_log10"]
        for c in dpu_needed:
            if c not in tmp.columns:
                raise ValueError(f"Need {c} in outcomes_df (from the CSV writer).")

        dpu = tmp.copy()
        dpu_dist_l = dpu["which_distribution"].astype(str).str.lower()
        dpu_is_yield = dpu_dist_l.map(is_yield_dist)
        dpu_is_excluded = dpu_dist_l.isin(EXCLUDE_DPU_DISTS)

        dpu = dpu[~dpu_is_yield & ~dpu_is_excluded].copy()
        if USE_OUTCOME_TARGET_RATS:
            # Use ratio when both outcome and target exist, otherwise use per-dollar outcome
            dpu["_val"] = np.where(
                dpu["outcome_over_target"].notna(),
                dpu["outcome_over_target"],
                dpu["outcome_norm_dollars_per_unit_log10"],
            )
            n_ratio = dpu["outcome_over_target"].notna().sum()
            n_total_before = len(dpu)
            print(
                f"[USE_OUTCOME_TARGET_RATS] DPU: Using ratio for {n_ratio}/{n_total_before} rows ({100*n_ratio/n_total_before:.1f}%)"
            )
        elif USE_RATIO_FOR_SPECIFIC_DISTS_ONLY:
            # Only use ratio for specific distributions
            dist_lower = dpu["which_distribution"].astype(str).str.lower()
            use_ratio = dpu["outcome_over_target"].notna() & dist_lower.isin(
                [d.lower() for d in RATIO_ONLY_DISTRIBUTIONS]
            )
            dpu["_val"] = np.where(
                use_ratio,
                dpu["outcome_over_target"],
                dpu["outcome_norm_dollars_per_unit_log10"],
            )
            n_ratio = use_ratio.sum()
            n_total_before = len(dpu)
            print(
                f"[USE_RATIO_FOR_SPECIFIC_DISTS_ONLY] DPU: Using ratio for {n_ratio}/{n_total_before} rows ({100*n_ratio/n_total_before:.1f}%)"
            )
        else:
            dpu["_val"] = dpu["outcome_norm_dollars_per_unit_log10"]
        dpu = dpu[dpu["_val"].notna()].copy()
        dpu["colname"] = dpu.apply(
            lambda r: f"out_dpu_{_safe(r['which_distribution'])}__{_safe(r['which_units'])}",
            axis=1,
        )

        meta_dpu = (
            dpu[["colname", "which_distribution", "which_units"]]
            .drop_duplicates()
            .set_index("colname")[["which_distribution", "which_units"]]
            .to_dict(orient="index")
        )

        wide_dpu = dpu.pivot_table(
            index="activity_id", columns="colname", values="_val", aggfunc="mean"
        ).sort_index()

    print("\n[DPU DEBUG] USE_SPLIT =", USE_SPLIT)
    print(
        "[DPU DEBUG] source _val col =",
        (
            "outcome_norm_dollars_per_unit_split_log10"
            if USE_SPLIT
            else "outcome_norm_dollars_per_unit_log10"
        ),
    )

    # ---- row-level stats before pivot (on dpu dataframe) ----
    print("\n[DPU DEBUG] long rows kept:", len(dpu))
    print("[DPU DEBUG] unique activity_id:", dpu["activity_id"].nunique())
    print(
        "[DPU DEBUG] unique (dist, units):",
        dpu[["which_distribution", "which_units"]].drop_duplicates().shape[0],
    )

    v = dpu["_val"].astype(float)
    print("\n[DPU DEBUG] _val describe:")
    print(v.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    # top groups by count + their _val summaries
    g = dpu.groupby(["which_distribution", "which_units"])["_val"]
    grp_counts = g.size().sort_values(ascending=False)
    print("\n[DPU DEBUG] top (dist,units) by row count:")
    print(grp_counts.head(30).to_string())

    top_pairs = grp_counts.head(10).index.tolist()
    if top_pairs:
        print("\n[DPU DEBUG] top 10 (dist,units) value stats:")
        for dist, units in top_pairs:
            s = g.get_group((dist, units)).astype(float)
            print(
                f"  - {dist} / {units}: n={len(s)}  mean={s.mean():.4g}  sd={s.std(ddof=0):.4g}  p50={s.median():.4g}  p95={s.quantile(0.95):.4g}  p99={s.quantile(0.99):.4g}"
            )

    # ---- wide stats after pivot ----
    print("\n[DPU DEBUG] wide_dpu shape:", wide_dpu.shape)  # (n_activities, n_outcomes)
    nn_col = wide_dpu.notna().sum(axis=0).astype(int).sort_values(ascending=False)
    print("\n[DPU DEBUG] non-null per outcome col (top 40):")
    print(nn_col.head(40).to_string())

    nn_row = wide_dpu.notna().sum(axis=1).astype(int)
    print("\n[DPU DEBUG] outcomes-per-activity distribution:")
    print(nn_row.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    print("[DPU DEBUG] activities with 0 dpu outcomes:", int((nn_row == 0).sum()))

    # ---- "how zagg might aggregate" on this wide block ----
    # zagg long-form creates one training row per (activity_id, outcome_col) that is notna.
    # Then (in the wide helper) averaging per-activity across available outcomes is common.
    # This mimics that aggregation (within DPU only) in two ways:
    #   (A) raw mean of available DPU outcomes per activity
    #   (B) mean of per-column z-scores per activity (using *THIS* universe stats; later TRAIN-only)
    mu = wide_dpu.mean(axis=0, skipna=True)
    sd = wide_dpu.std(axis=0, ddof=0, skipna=True).replace(0.0, np.nan)

    raw_mean_per_act = wide_dpu.mean(axis=1, skipna=True)
    zmean_per_act = ((wide_dpu - mu) / sd).mean(axis=1, skipna=True)

    print("\n[DPU DEBUG] per-activity raw mean(_val across DPU cols) describe:")
    print(
        raw_mean_per_act.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string()
    )

    print(
        "\n[DPU DEBUG] per-activity mean(z) across DPU cols describe (universe stats):"
    )
    print(zmean_per_act.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    # show the activities that dominate the aggregation (many outcomes)
    print(
        "\n[DPU DEBUG] activities with most DPU outcomes (these dominate any simple averaging):"
    )
    print(nn_row.sort_values(ascending=False).head(25).to_string())

    dpu_cols = list(wide_dpu.columns.astype(str))

    print("[outcomes] DPU wide:", wide_dpu.shape, "cols:", len(dpu_cols))
    print("[outcomes] DPU excluded yield +", EXCLUDE_DPU_DISTS)

    return wide_raw, raw_cols, meta_raw, wide_dpu, dpu_cols, meta_dpu


def train_rfs_for_outcomes(
    *,
    data: pd.DataFrame,
    feature_cols,
    outcome_cols,
    train_idx,
    rf_params,
    min_train=30,
    pred_prefix="pred_",
    return_models: bool = False,
):
    trained = []
    models = {}
    runner_used = {}  # out_col -> "rf" or "ridge"

    print("\n=== TRAINING PER-OUTCOME RANDOM FORESTS ===")
    print("min_train:", min_train)
    print("n_outcome_cols:", len(outcome_cols))

    for out_col in outcome_cols:
        train_idx_t = pd.Index(train_idx).intersection(
            data.index[data[out_col].notna()]
        )
        n = len(train_idx_t)

        if n < min_train:
            print(f"SKIP {out_col}: only {n} labeled TRAIN rows (<{min_train})")
            continue

        print(f"TRAIN {out_col}: labeled TRAIN rows={n}")

        if out_col in [
            "out_dpu_beneficiaries__people",
            "out_rating",
            "out_raw_benefit_cost_ratios__ratio",
            "out_raw_generation_capacity__mw",
            "out_raw_trees_planted__count",
            "out_raw_water_connections__connections",
            "out_dpu_water_connections__connections",
        ]:

            yhat_all, model = run_random_forest_median_impute_noclip(
                data=data,
                feature_cols=feature_cols,
                target_col=out_col,
                train_index=train_idx_t,
                rf_params=rf_params,
                ensemble_with_extratrees=True,
            )
            runner_used[out_col] = "rf"

        elif out_col in [
            "out_dpu_co2_emission_reductions__tonnes_co2e",
            "out_dpu_generation_capacity__mw",
            "out_raw_beneficiaries__people",
            "out_raw_co2_emission_reductions__tonnes_co2e",
            "out_raw_economic_rate_of_return__percent",
            "out_raw_financial_rate_of_return__percent",
            "out_raw_yield_increases_percent__percent",
        ]:
            yhat_all, model = run_ridge_glm_median_impute_noclip(
                data=data,
                feature_cols=feature_cols,
                target_col=out_col,
                train_index=train_idx_t,
            )
            runner_used[out_col] = "ridge"

        else:
            yhat_all, model = run_random_forest_median_impute_noclip(
                data=data,
                feature_cols=feature_cols,
                target_col=out_col,
                train_index=train_idx_t,
                rf_params=rf_params,
                ensemble_with_extratrees=True,
            )
            runner_used[out_col] = "rf"

        pred_col = f"{pred_prefix}{out_col}"
        data[pred_col] = pd.Series(yhat_all, index=data.index)

        trained.append(out_col)
        if return_models:
            models[out_col] = model

    print("\nTrained outcome models:", len(trained))

    if return_models:
        return trained, models, runner_used
    return trained


from sklearn.inspection import permutation_importance

# pairwise_ordering_prob and spearman_correlation now imported from utils/scoring_metrics.py


def evaluate_outcomes_once_per_activity(
    *,
    data: pd.DataFrame,
    test_idx,
    outcome_cols,
    pred_prefix="pred_",
    min_outcomes_per_activity=1,
    print_worst=15,
):
    test_idx = pd.Index(test_idx).astype(str)

    pred_cols = [f"{pred_prefix}{c}" for c in outcome_cols]
    have = [c for c, p in zip(outcome_cols, pred_cols) if p in data.columns]
    outcome_cols = have
    pred_cols = [f"{pred_prefix}{c}" for c in outcome_cols]
    if not outcome_cols:
        raise ValueError("No outcome_cols have matching prediction columns in data.")

    y_true = data.loc[test_idx, outcome_cols].astype(float)
    y_pred = data.loc[test_idx, pred_cols].astype(float)
    y_pred.columns = y_true.columns

    err = y_pred - y_true

    n = err.notna().sum(axis=1)
    rmse = np.sqrt((err**2).mean(axis=1, skipna=True))
    mae = err.abs().mean(axis=1, skipna=True)
    bias = err.mean(axis=1, skipna=True)

    # collapse each activity to one target + one pred (mean over available outcomes)
    y_true_mean = y_true.mean(axis=1, skipna=True)
    y_pred_mean = y_pred.mean(axis=1, skipna=True)

    out = pd.DataFrame(
        {
            "n_outcomes": n,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "y_true_mean": y_true_mean,
            "y_pred_mean": y_pred_mean,
        },
        index=test_idx,
    )

    out = out[out["n_outcomes"] >= min_outcomes_per_activity].copy()

    print("\n=== PER-ACTIVITY OUTCOME EVAL (TEST) ===")
    print(f"kept activities: {len(out)} (n_outcomes >= {min_outcomes_per_activity})")

    # ---- R^2 metrics ----
    # (A) per-activity R^2 (each activity weight = 1)
    r2_activity = r2_score(out["y_true_mean"], out["y_pred_mean"])
    print(f"\nR^2 (per-activity mean): {r2_activity:.6g}")

    # (B) cell-wise R^2 (each outcome-cell counts)
    yt = y_true.values
    yp = y_pred.values
    mask = np.isfinite(yt) & np.isfinite(yp)
    r2_cell = r2_score(yt[mask], yp[mask])
    print(f"R^2 (cell-wise):        {r2_cell:.6g}")

    # keep existing summaries
    overall_rmse = float(np.sqrt(np.nanmean((yp - yt) ** 2)))
    overall_mae = float(np.nanmean(np.abs(yp - yt)))
    print(f"\nRMSE (cell-wise): {overall_rmse:.6g}")
    print(f"MAE  (cell-wise): {overall_mae:.6g}")

    # ---- Bootstrap CIs for R^2, Pairwise, and Spearman ----
    # Use cell-wise (flattened) values for all metrics
    yt_flat = yt[mask]
    yp_flat = yp[mask]

    # R^2 bootstrap CI
    r2_ci = bootstrap_ci(
        yt_flat, yp_flat, lambda y_t, y_p: r2_score(y_t, y_p), n_bootstrap=1000
    )
    print(f"\nR^2 (cell-wise) 95% CI: [{r2_ci['lower']:.4f}, {r2_ci['upper']:.4f}]")

    # Pairwise ordering probability
    pairwise_score = pairwise_ordering_prob(yt_flat, yp_flat)
    pairwise_ci = bootstrap_ci(
        yt_flat, yp_flat, pairwise_ordering_prob, n_bootstrap=1000
    )
    print(
        f"Pairwise ordering: {pairwise_score:.4f}  95% CI: [{pairwise_ci['lower']:.4f}, {pairwise_ci['upper']:.4f}]"
    )
    print(
        f"DEBUG MISSING Prediction variance: {np.var(yp_flat):.6f}, range: [{np.min(yp_flat):.3f}, {np.max(yp_flat):.3f}]"
    )

    # Spearman correlation
    spearman_score = spearman_correlation(yt_flat, yp_flat)
    spearman_ci = bootstrap_ci(yt_flat, yp_flat, spearman_correlation, n_bootstrap=1000)
    print(
        f"Spearman correlation: {spearman_score:.4f}  95% CI: [{spearman_ci['lower']:.4f}, {spearman_ci['upper']:.4f}]"
    )

    if print_worst and len(out) > 0:
        worst = out.sort_values("rmse", ascending=False).head(print_worst)
        print(f"\nWorst {len(worst)} activities by per-activity RMSE:")
        print(
            worst[
                ["n_outcomes", "rmse", "mae", "bias", "y_true_mean", "y_pred_mean"]
            ].to_string()
        )

    metrics = {
        "r2_activity": float(r2_activity),
        "r2_cell": float(r2_cell),
        "r2_cell_ci_lower": r2_ci["lower"],
        "r2_cell_ci_upper": r2_ci["upper"],
        "pairwise": float(pairwise_score),
        "pairwise_ci_lower": pairwise_ci["lower"],
        "pairwise_ci_upper": pairwise_ci["upper"],
        "spearman": float(spearman_score),
        "spearman_ci_lower": spearman_ci["lower"],
        "spearman_ci_upper": spearman_ci["upper"],
        "rmse_cell": float(overall_rmse),
        "mae_cell": float(overall_mae),
        "n_activities": int(len(out)),
    }

    # ---- largest true value anywhere in y_true (across all outcomes + activities) ----
    s = y_true.stack(dropna=True)  # MultiIndex: (activity_id, outcome_col) -> value
    aid_max, col_max = s.idxmax()
    val_max = float(s.loc[(aid_max, col_max)])

    print("\n=== MAX TRUE VALUE IN TEST SET ===")
    print("activity_id:", aid_max)
    print("outcome_col:", col_max)
    print("true_value:", val_max)

    print("\n--- full activity row from `data` ---")
    print(data.loc[aid_max].to_string())

    print("\n--- all true outcomes for that activity (wide) ---")
    print(y_true.loc[aid_max].sort_values(ascending=False).to_string())

    # optional: show the predicted value for the same cell if available
    pred_colname = f"{pred_prefix}{col_max}"
    if pred_colname in data.columns:
        print("\n--- pred for that outcome ---")
        print("pred_value:", float(data.loc[aid_max, pred_colname]))

    return out, metrics


def load_predictions_from_jsonl(filepath, parser, series_name):
    """
    Generic loader: JSONL -> Series(activity_id -> int rating 0..5)
    using a parser(content, record) -> int or None.

    Supports:
      - ChatGPT-style records: {"response": {"content": "..."}}
      - Gemini-style records: {"response_text": "..."}
    """
    preds = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            activity_id = data.get("activity_id")
            if activity_id is None:
                continue

            # 1) Try ChatGPT-style
            content = None
            resp = data.get("response")
            if isinstance(resp, dict):
                content = resp.get("content") or resp.get("text")

            # 2) Fall back to plain response_text
            if not content:
                content = data.get("response_text")

            if not content:
                # nothing to parse
                continue

            value = parser(content, data)
            if value is None:
                continue

            preds[activity_id] = value

    return pd.Series(preds, name=series_name)


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


split_latest_by_date_with_cutoff = _split_canonical


def only_use_these_and_assert_exist_with_counts(
    data: pd.DataFrame,
    only_cols: list[str],
    *,
    min_nonnull: int = 1,
    label: str = "ALL",
) -> list[str]:
    only_cols = list(only_cols)

    missing = [c for c in only_cols if c not in data.columns]
    if missing:
        print(f"\n[{label}] MISSING ({len(missing)}):")
        for c in missing:
            print("  -", c)

        # helpful context: show similarly-named columns that DO exist
        have = pd.Index(map(str, data.columns))
        for c in missing:
            prefix = c.split("__", 1)[0]  # e.g. out_raw_blah
            matches = [h for h in have if h.startswith(prefix)]
            if matches:
                print(
                    f"\n[{label}] existing cols starting with '{prefix}' (showing up to 30):"
                )
                for h in matches[:30]:
                    print("  +", h)

        raise ValueError(f"[{label}] missing outcome cols ({len(missing)})")

    counts = data[only_cols].notna().sum().astype(int).sort_values(ascending=False)
    print(f"\n[{label}] outcome non-null counts (n_rows with value):")
    print(counts.to_string())

    # ---- rows with zero selected outcomes ----
    row_has_any = data[only_cols].notna().any(axis=1)
    n_empty = int((~row_has_any).sum())

    too_small = counts[counts < int(min_nonnull)]
    if len(too_small):
        print(f"\n[{label}] TOO SMALL (<{min_nonnull})")
        print(too_small.to_string())
        print(f"[{label}] some outcome cols have <{min_nonnull} non-null rows")

    return only_cols


def _latex_escape(s: str) -> str:
    """Minimal escape for LaTeX table text."""
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


def print_latex_outcome_table(df_scores: pd.DataFrame) -> None:
    """
    Print a LaTeX table of outcome model performance with confidence intervals.

    Columns: Outcome (with units), R^2 (with CI), Pairwise (with CI),
             MAE, N_train/N_val/N_test
    """
    print("\n" + "=" * 80)
    print("=== LATEX TABLE ===")
    print("=" * 80)

    df = df_scores.copy()
    df = df.sort_values("r2_activity", ascending=False)

    def fmt_with_ci(value, ci_lower, ci_upper, decimals=2):
        """Format as 'value [lower, upper]'"""
        if pd.isna(value):
            return ""
        val_str = f"{value:.{decimals}f}"
        if pd.isna(ci_lower) or pd.isna(ci_upper):
            return val_str
        ci_str = f"[{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"
        return f"{val_str} {ci_str}"

    def fmt_val(value, decimals=2):
        if pd.isna(value):
            return ""
        return f"{value:.{decimals}f}"

    def fmt_int(value):
        if pd.isna(value):
            return ""
        return str(int(value))

    # Map outcome names to readable names
    def format_outcome_name(row):
        """Generate readable outcome name with units at the end."""
        outcome = row.get("outcome_col", "")

        # Special case for zagg
        if outcome == "outcomes_z_activity_mean":
            return "All selected outcomes (z-aggregate activity mean)"

        # Try to clean up the outcome name
        outcome_clean = outcome.replace("out_raw_", "").replace("out_dpu_", "")
        outcome_clean = outcome_clean.replace("_", " ").title()

        # Get units and append to outcome name
        units = format_units(row)
        if units:
            return f"{outcome_clean} ({units})"

        return outcome_clean

    # Map units
    def format_units(row):
        """Extract units from outcome column name."""
        units = row.get("which_units", "")
        outcome = row.get("outcome_col", "")

        if units == "z":
            return "z"

        # Try to extract from outcome name (e.g., out_dpu_beneficiaries__people)
        if "__" in outcome:
            unit_part = outcome.split("__")[-1]
            return unit_part.replace("_", " ")

        return units if units else ""

    # Map transformations
    def format_transform(row):
        """Describe the transformation applied."""
        dist = row.get("which_distribution", "")
        units = row.get("which_units", "")

        if units == "z":
            return "none (z-score)"
        if dist == "log":
            return r"$\log$"
        return "none"

    # Start LaTeX table
    print(r"\begin{table}")
    print(
        r"\caption{Predictive performance of cost-effectiveness outcome models on the validation set. "
        + r"R$^2$ and Pairwise ordering probability are shown with "
        + r"95\% bootstrap confidence intervals. Outcomes are sorted by R$^2$ (descending).}"
    )
    print(r"\label{tab:outcome_model_performance}")
    print()
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\renewcommand{\arraystretch}{1.15}")
    print()
    print(
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X "
        + r"r r r "
        + r">{\raggedleft\arraybackslash}p{2.5cm}@{}}"
    )
    print()
    print(r"\toprule")
    print(
        r"Outcome & $R^2$ & Pairwise & MAE & "
        + r"$N_{\mathrm{train}}/N_{\mathrm{val}}/N_{\mathrm{test}}$ \\"
    )
    print(r"\midrule")

    for _, row in df.iterrows():
        outcome_name = format_outcome_name(row)

        # Get metrics with CIs
        r2_str = fmt_with_ci(
            row.get("r2_activity"),
            row.get("r2_cell_ci_lower"),
            row.get("r2_cell_ci_upper"),
            decimals=2,
        )

        pairwise_str = fmt_with_ci(
            row.get("pairwise"),
            row.get("pairwise_ci_lower"),
            row.get("pairwise_ci_upper"),
            decimals=2,
        )

        mae_str = fmt_val(row.get("mae_cell"), decimals=2)

        # Get sample sizes - note the order: train/val/test (not val/train/test)
        n_train = fmt_int(row.get("n_train"))
        n_val = fmt_int(row.get("n_val"))
        n_test = fmt_int(
            row.get("n_test", row.get("n_held"))
        )  # use n_held if n_test not available
        n_str = f"{n_train} / {n_val} / {n_test}"

        # Escape special characters in outcome name
        outcome_escaped = _latex_escape(outcome_name)

        # Print row
        print(
            f"{outcome_escaped} & {r2_str} & "
            + f"{pairwise_str} & {mae_str} & {n_str} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabularx}")
    print(r"\end{table}")
    print()


def main():
    META_BY_COL = {}

    # Validate ratio configuration
    ratio_flags = [
        USE_OUTCOME_TARGET_RATS,
        INCLUDE_RATIO_IN_ZAGG,
        USE_RATIO_FOR_SPECIFIC_DISTS_ONLY,
    ]
    if sum(ratio_flags) > 1:
        raise ValueError(
            "Cannot use multiple ratio options simultaneously.\n"
            "  Option A (USE_OUTCOME_TARGET_RATS): Replaces ALL outcomes with ratio when both exist\n"
            "  Option B (INCLUDE_RATIO_IN_ZAGG): Adds ratio as additional outcome\n"
            "  Option C (USE_RATIO_FOR_SPECIFIC_DISTS_ONLY): Replaces ONLY specific distributions with ratio\n"
            "  Choose one or set all to False."
        )

    print("\n" + "=" * 80)
    print("ZAGG CONFIGURATION")
    print("=" * 80)
    print(f"USE_OUTCOME_TARGET_RATS: {USE_OUTCOME_TARGET_RATS}")
    print(f"INCLUDE_RATIO_IN_ZAGG: {INCLUDE_RATIO_IN_ZAGG}")
    print(f"USE_RATIO_FOR_SPECIFIC_DISTS_ONLY: {USE_RATIO_FOR_SPECIFIC_DISTS_ONLY}")
    if USE_RATIO_FOR_SPECIFIC_DISTS_ONLY:
        print(f"  RATIO_ONLY_DISTRIBUTIONS: {RATIO_ONLY_DISTRIBUTIONS}")
    print(f"MIN_TRAIN_ROWS_PER_GROUP: {MIN_TRAIN_ROWS_PER_GROUP}")
    print(f"MIN_TEST_ROWS_PER_GROUP: {MIN_TEST_ROWS_PER_GROUP}")
    print("=" * 80 + "\n")

    # ---------------------------------------------------------------------------
    # MAIN SCRIPT
    # ---------------------------------------------------------------------------
    print("\n=== Predicting outcomes ===")

    info_ids = (
        pd.read_csv(INFO_FOR_ACTIVITY_FORECASTING, usecols=["activity_id"])[
            "activity_id"
        ]
        .astype(str)
        .str.strip()
    )
    info_ids = pd.Index(info_ids)

    # outcomes wide

    outcomes_df = load_activity_outcomes_df(
        ACTIVITY_OUTCOMES_PATH,
        dedupe=True,
        require_value_cols=("outcome_norm",),
    )

    wide_raw, raw_cols, meta_raw, wide_dpu, dpu_cols, meta_dpu = (
        outcomes_to_wide_two_variants(outcomes_df)
    )
    wide_raw.index = wide_raw.index.astype(str)
    wide_dpu.index = wide_dpu.index.astype(str)
    wide_all = wide_raw.join(wide_dpu, how="outer")

    # ratings (Series indexed by activity_id)
    ratings = load_ratings(MERGED_OVERALL_RATINGS)
    ratings.index = ratings.index.astype(str)

    use_cols = [
        c for c in ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS if c in wide_all.columns
    ]

    wide_outcome_cols = use_cols[:]  # only the wide columns (raw/dpu)

    # targets that should be modeled as outcomes (wide + maybe rating)
    train_outcome_cols = wide_outcome_cols[:]
    # RATINGS EXCLUDED FROM TRAINING ENTIRELY

    # Base set is activities with at least one quantitative outcome.
    has_wide = wide_all.reindex(info_ids)[use_cols].notna().any(axis=1)
    has_rating = ratings.reindex(info_ids).notna()

    base_ids = info_ids[has_wide.values]

    data = pd.DataFrame(index=base_ids)
    data = data.join(ratings.rename("out_rating"), how="left")
    data = data.join(wide_all, how="left")

    dists = {"economic_rate_of_return", "financial_rate_of_return"}

    sub = outcomes_df[
        outcomes_df["which_distribution"].astype(str).str.lower().isin(dists)
    ].copy()

    scores = {}  # (dist, units, variant) -> dict of metrics
    all_imps = {}  # (variant, outcome_col) -> importance df

    if not USE_OUTCOME_TARGET_RATS:

        # ---- after: data = wide_raw.join(wide_dpu, how="outer") and ONLY_USE... defined ----
        outcome_cols = only_use_these_and_assert_exist_with_counts(
            data,
            use_cols,
            min_nonnull=1,  # bump to enforce coverage
            label="UNIVERSE",
        )

        raw_cols = [c for c in wide_outcome_cols if c.startswith("out_raw_")]
        dpu_cols = [c for c in wide_outcome_cols if c.startswith("out_dpu_")]

        META_BY_COL.update(meta_raw)
        META_BY_COL.update(meta_dpu)
        META_BY_COL = {k: v for k, v in META_BY_COL.items() if k in wide_outcome_cols}

        if "out_rating" in train_outcome_cols:
            META_BY_COL["out_rating"] = {
                "which_distribution": "rating",
                "which_units": "rating",
            }

    # outcome_cols = raw_cols + dpu_cols  # <- what the loop iterates over

    # ---- Include ratio in zagg aggregate (alongside other outcomes) ----
    if INCLUDE_RATIO_IN_ZAGG and not USE_OUTCOME_TARGET_RATS:
        print("\n=== INCLUDE_RATIO_IN_ZAGG is True ===")

        # Load ratio data (average per activity)
        ratio_wide_for_zagg = (
            outcomes_df[["activity_id", "outcome_over_target"]]
            .dropna(subset=["outcome_over_target"])
            .groupby("activity_id")["outcome_over_target"]
            .mean()
            .to_frame(
                "out_ratio__outcome_over_target"
            )  # Add "out_" prefix for consistency
        )

        # Join to data (left join - activities without ratio will have NaN)
        data = data.join(ratio_wide_for_zagg, how="left")

        # Add to train_outcome_cols so it becomes part of zagg
        if "out_ratio__outcome_over_target" in data.columns:
            train_outcome_cols.append("out_ratio__outcome_over_target")
            wide_outcome_cols.append("out_ratio__outcome_over_target")

            # Add metadata for the ratio
            META_BY_COL["out_ratio__outcome_over_target"] = {
                "which_distribution": "outcome_over_target",
                "which_units": "ratio",
            }

            print(f"  Added out_ratio__outcome_over_target to train_outcome_cols")
            print(
                f"  Activities with ratio: {data['out_ratio__outcome_over_target'].notna().sum()} / {len(data)}"
            )
        else:
            print("  WARNING: out_ratio__outcome_over_target not found in data columns")

        # keep_outcomes = outcome_cols  # not the raw constant; already filtered to existing cols
        # data = data[["out_rating"] + keep_outcomes]

    if USE_CACHE and Path(CACHE_FILE).exists():
        print(f"\n=== LOADING FROM CACHE: {CACHE_FILE} ===")
        with open(CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)

        data_to_merge = cache_data["data"]
    else:
        print("Loading grades data...")
        grades_df = load_grades(ALL_GRADES)

        print("Loading activity scope...")
        scope_df = load_activity_scope(INFO_FOR_ACTIVITY_FORECASTING)

        print("Loading implementing_org_type...")
        impl_type_df = load_implementing_org_type(INFO_FOR_ACTIVITY_FORECASTING)

        print("Loading gdp_percap...")
        gdp_df = load_gdp_percap(INFO_FOR_ACTIVITY_FORECASTING)

        print("Loading planned expenditure...")
        expend_df = return_misc_disbursement_or_planned_disbursement(
            INFO_FOR_ACTIVITY_FORECASTING, OUT_MISC
        )

        print("Loading world bank indicators...")
        world_bank_df = load_world_bank_indicators(INFO_FOR_ACTIVITY_FORECASTING)

        print("Loading is_completed...")
        is_completed = load_is_completed(INFO_FOR_ACTIVITY_FORECASTING)

        print("Loading loan_or_disbursement...")
        lod_df = load_loan_or_disbursement()

        # outcome_cols = outcome_cols + ["out_rating"]
        # outcome_cols = ["out_rating"]

        print(
            "Loading targets/context maps features (sector dummies, umap2, text_len, targets number count)..."
        )
        tc_maps_df = load_targets_context_maps_features(TARGETS_CONTEXT_MAPS)

        print("Loading sector cluster allocations (HHI, cluster features)...")
        sector_clusters_df = data_sector_clusters(FINANCE_SECTORS)

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

        print("outcome_cols")
        print(outcome_cols)

        print("\nMerging datasets...")
        data_to_merge = pd.DataFrame(index=base_ids)

        data_to_merge = data_to_merge.join(is_completed, how="left")
        data_to_merge = data_to_merge.join(grades_df, how="left")
        data_to_merge = data_to_merge.join(scope_df, how="left")
        data_to_merge = data_to_merge.join(gdp_df, how="left")
        data_to_merge = data_to_merge.join(expend_df, how="left")
        data_to_merge = data_to_merge.join(world_bank_df, how="left")
        data_to_merge = data_to_merge.join(lod_df, how="left")
        data_to_merge = data_to_merge.join(impl_type_df, how="left")
        data_to_merge = data_to_merge.join(tc_maps_df, how="left")

        data_to_merge = data_to_merge.join(regions_df, how="left")
        data_to_merge = data_to_merge.join(
            sector_clusters_df, how="left"
        )  # sector HHI and cluster allocations

        # Encode activity_scope
        data_to_merge["activity_scope"] = pd.to_numeric(
            data_to_merge["activity_scope"], errors="coerce"
        )
        # Features / labels

        if USE_CACHE:
            print(f"\n=== SAVING TO CACHE: {CACHE_FILE} ===")
            CACHE_FILE.parent.mkdir(
                parents=True, exist_ok=True
            )  # Create parent directory if needed
            cache_data = {
                "data": data_to_merge,
            }
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(cache_data, f, protocol=4)

    data = data.join(data_to_merge, how="left")

    data, sim_feature_cols = add_similarity_features(
        data, INFO_FOR_ACTIVITY_FORECASTING, KEEP_REPORTING_ORGS
    )

    org_cols = [c for c in data.columns if c.startswith("rep_org_")]

    if RESTRICT_TO_REPORTING_ORGS:
        data = restrict_to_reporting_orgs_exact(data, KEEP_REPORTING_ORGS)

    s = data["reporting_orgs"].fillna("").astype(str).str.strip()
    vc = s.value_counts(dropna=False)

    data = add_dates_to_dataframe(data, INFO_FOR_ACTIVITY_FORECASTING)

    # Load country_location for WDI matching
    country_df = pd.read_csv(
        INFO_FOR_ACTIVITY_FORECASTING, usecols=["activity_id", "country_location"]
    )
    country_df = country_df.set_index("activity_id")
    data = data.join(country_df, how="left")

    data["has_start"] = data["start_date"].notna()

    base = data.copy()

    # require the split keys exist
    base = base[base["start_date"].notna() & base["reporting_orgs"].notna()].copy()

    if DROP_INCOMPLETE:
        base = base[base["is_completed"].fillna(0).astype(int) == 1].copy()

    train_idx, val_idx, held_idx = split_latest_by_date_with_cutoff(
        base, date_col="start_date"
    )

    # Save split IDs to the shared eval-set-sizes directory for cross-script comparison
    _eval_dir = Path("../../data/eval_set_sizes")
    _eval_dir.mkdir(parents=True, exist_ok=True)
    _ce_splits = pd.concat(
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
    _ce_splits.to_csv(_eval_dir / "cost_effectiveness_splits.csv", index=False)
    print(
        f"[eval_set_sizes] cost_effectiveness splits saved to {_eval_dir / 'cost_effectiveness_splits.csv'}"
    )

    # optional: show a few ids
    both_ids = info_ids[(has_wide & has_rating).values]

    dups_in_data = pd.Index(data.columns).astype(str)
    dups_in_data = dups_in_data[dups_in_data.duplicated()].unique().tolist()

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
        # 'is_completed', # I think this is info that would not be available at start of project.
        # 'finance_label',
        "finance_is_loan",
        # 'finance_class_method',
        # 'rating',       # lol let's not include the rating!
        # 'rep_org_main_idx',
        # 'start_date_num',
        "start_date",  # needed for WDI matching
        "country_location",  # needed for WDI matching
        "planned_duration",
        "planned_expenditure",
        # "is_government_impl",
        # "is_ngo_impl",
        # "planned_loan",
        *[
            f"rep_org_{i}" for i in range(NUM_ORGS_KEEP - 1)
        ],  # helps a lot, that "- 1". reduces colinearity -> 0.04 R^2 improvement
        # ---- new features from outputs_targets_context_maps.jsonl ----
        "region_AFE",
        "region_AFW",
        "region_EAP",
        "region_ECA",
        "region_LAC",
        "region_MENA",
        "region_SAS",
        "umap3_x",
        "umap3_y",
        "umap3_z",
        # "text_len_chars",
        # "targets_number_count",
        # "sector_distance",
        # "country_distance",
        # *[c for c in data.columns if c.startswith("env_cat_")],
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

    if USE_RATING_AS_FEATURE_AND_DONT_PREDICT_IT:
        feature_cols.append("out_rating")
        # pass

    missing_feats = [c for c in feature_cols if c not in data.columns]
    print("missing features:", missing_feats[:50])
    if missing_feats:
        feature_cols = [c for c in feature_cols if c not in missing_feats]

    feat_idx = pd.Index(feature_cols)
    dups = feat_idx[feat_idx.duplicated()].unique().tolist()
    if dups:
        raise ValueError(f"Duplicate feature names in feature_cols: {dups}")

    if USE_SAME_FEATURES_AS_OVERALL_RATING:
        _wgi_cols = [
            "wgi_control_of_corruption_est",
            "wgi_government_effectiveness_est",
            "wgi_political_stability_est",
            "wgi_regulatory_quality_est",
            "wgi_rule_of_law_est",
        ]
        data["governance_composite"] = data[_wgi_cols].mean(axis=1)
        data["log_planned_expenditure"] = data["planned_expenditure"]
        data["planned_expenditure"] = np.exp(data["planned_expenditure"])
        data["expenditure_x_complexity"] = (
            data["planned_expenditure"] * data["complexity"]
        )
        data["expenditure_per_year_log"] = np.log(
            (data["planned_expenditure"] / data["planned_duration"]).where(
                (data["planned_duration"] >= 1)
                & (data["planned_expenditure"] >= 100000),
                np.nan,
            )
        )
        feature_cols = [f for f in feature_cols if f not in _wgi_cols]
        feature_cols += [
            "sector_distance",
            "country_distance",
            "governance_composite",
            "expenditure_x_complexity",
            "expenditure_per_year_log",
            "log_planned_expenditure",
        ]
        feature_cols = [f for f in feature_cols if f in data.columns]

    # Random Forest parameters (used for zagg model)
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

    feature_cols = [
        c for c in feature_cols if c not in ["start_date", "country_location"]
    ]

    if True:  # Always run zagg pipeline (old: if not USE_OUTCOME_TARGET_RATS)

        # only use the explicitly whitelisted outcome cols
        use_cols = [
            c for c in ONLY_USE_THESE_AND_ASSERT_EXIST_WITH_COUNTS if c in data.columns
        ]
        if not use_cols:
            raise ValueError(
                "No ONLY_USE_THESE... cols found in `data` at long-form stage."
            )

        train_outcomes = use_cols.copy()
        if USE_RATING_AS_FEATURE_AND_DONT_PREDICT_IT:
            if not ("out_rating" in data.columns):
                print("ERROR! using rating as freature but doesn't exist.")
                sys.exit(1)
        # out_rating is not a cost-effectiveness outcome — never include it in zagg

        # keep the activity if at least one of the train_outcomes columns is non-NA. It does not require all (or even most) to be present
        data_raw_univ = data[data[train_outcomes].notna().any(axis=1)].copy()

        if DROP_INCOMPLETE:
            complete_mask = data_raw_univ["is_completed"].fillna(0).astype(int) == 1
            data_raw_univ = data_raw_univ.loc[complete_mask].copy()

        # split by activity_id (NO leakage across groups)
        train_aids, val_aids, held_aids = split_activity_ids_by_date_with_cutoff(
            data_raw_univ, date_col="start_date"
        )
        test_aids = val_aids if not USE_VAL_IN_TRAIN else held_aids
        plot_aids_tv = train_aids.union(val_aids).union(held_aids)  # train + val

        use_cols_nonrating = [c for c in use_cols if c != "out_rating"]

        # build long rows only from selected outcomes
        long_df = build_long_outcome_rows(
            data_raw_univ,
            outcome_cols=train_outcome_cols,
            feature_cols=feature_cols,
            meta_by_col=META_BY_COL,
            group_col_name="which_group",
        )

        if INCLUDE_RATIO_IN_ZAGG and REMOVE_RATIO_FROM_VAL_AND_TEST:
            ratio_group = "out_ratio__outcome_over_target"

            # Count before removal
            ratio_train_before = (
                (long_df["which_group"] == ratio_group)
                & long_df["activity_id"].isin(train_aids)
            ).sum()
            ratio_test_before = (
                (long_df["which_group"] == ratio_group)
                & long_df["activity_id"].isin(test_aids)
            ).sum()

            # Remove ratio rows from test set
            ratio_test_mask = (long_df["which_group"] == ratio_group) & long_df[
                "activity_id"
            ].isin(test_aids)
            long_df = long_df[~ratio_test_mask].copy()

            print(f"\n=== Ratio filtering (train only) ===")
            print(f"  Ratio rows in TRAIN: {ratio_train_before} (kept)")
            print(f"  Ratio rows in TEST: {ratio_test_before} (removed)")
            print(f"  Total long_df rows after filter: {len(long_df)}")

        # ---- PARANOID: split overlap check (THIS IS THE ONLY SPLIT zagg uses) ----
        tr = set(map(str, train_aids))
        te = set(map(str, test_aids))
        ov = sorted(tr.intersection(te))
        assert len(ov) == 0, "LEAK: train/test activity_id overlap"

        # enforce minimum rows in TRAIN and TEST per group for reliable z-score calculation
        long_df, kept_groups = filter_groups_min_counts(
            long_df,
            train_aids=train_aids,
            test_aids=test_aids,
            group_col="which_group",
            min_train=MIN_TRAIN_ROWS_PER_GROUP,
            min_test=MIN_TEST_ROWS_PER_GROUP,
        )

        # group-wise z using TRAIN-only stats
        long_df = add_groupwise_z(
            long_df,
            train_aids=train_aids,
            group_col="which_group",
            y_col="y_raw",
            z_col="y_z",
        )

        print("\n=== GROUPWISE Z AUDIT ===")
        g = long_df["which_group"]

        for grp in g.unique():
            tr = long_df["activity_id"].isin(train_aids) & (g == grp)
            te = long_df["activity_id"].isin(test_aids) & (g == grp)
            mu_tr = float(long_df.loc[tr, "y_raw"].mean())
            sd_tr = float(long_df.loc[tr, "y_raw"].std(ddof=0))
            mu_te = float(long_df.loc[te, "y_raw"].mean())
            print(
                f"{grp:40s} train_mu={mu_tr: .4g} train_sd={sd_tr: .4g} test_mu={mu_te: .4g} train_n={int(tr.sum())} test_n={int(te.sum())}"
            )

        # Zagg row count diagnostic
        print("\n=== ZAGG ROW COUNT DIAGNOSTIC ===")
        train_mask = long_df["activity_id"].isin(train_aids)
        test_mask = long_df["activity_id"].isin(test_aids)

        print(f"Total long_df rows: {len(long_df)}")
        print(f"  Train rows: {train_mask.sum()}")
        print(f"  Test rows: {test_mask.sum()}")
        print(f"Unique activities: {long_df['activity_id'].nunique()}")
        print(f"  Train activities: {long_df.loc[train_mask, 'activity_id'].nunique()}")
        print(f"  Test activities: {long_df.loc[test_mask, 'activity_id'].nunique()}")
        print(f"Unique outcome groups: {long_df['which_group'].nunique()}")

        # Rows per activity stats (for zagg averaging)
        rows_per_act = long_df.groupby("activity_id").size()
        print(f"\nRows per activity (for zagg aggregation):")
        print(f"  Mean: {rows_per_act.mean():.1f}")
        print(f"  Median: {rows_per_act.median():.0f}")
        print(f"  Min: {rows_per_act.min()}")
        print(f"  Max: {rows_per_act.max()}")

        pprint.pprint(long_df)

        # split-specific composition
        test_mask = long_df["activity_id"].isin(set(map(str, test_aids)))

        # rows per activity in TEST (this drives zagg averaging)
        k = (
            long_df.loc[test_mask]
            .groupby("activity_id")
            .size()
            .sort_values(ascending=False)
        )

        # add group dummies as features
        long_df, grp_cols = add_group_dummies(
            long_df, group_col="which_group", prefix="grp"
        )

        long_feature_cols = feature_cols + grp_cols
        print(f"  Final feature count: {len(long_feature_cols)}")

        if ADD_INTERACTION_FEATURES:
            ####  ADD INTERACTION FEATURES ####
            top_features = [
                "umap3_z",
                "sector_cluster_Urban_flood_protection",
                "umap3_y",
                "umap3_x",
                "expenditure_x_complexity",
                "governance_x_complexity",
                "sector_cluster_Improved_transport_infrastructure",
                "gdp_x_duration",
                "gdp_percap",
                "governance_composite",
            ]

            # Create interaction features
            interactions = []
            for i in range(len(top_features)):
                for j in range(
                    i + 1, min(i + 4, len(top_features))
                ):  # Limit to avoid explosion
                    feat1, feat2 = top_features[i], top_features[j]
                    if feat1 in long_df.columns and feat2 in long_df.columns:
                        inter_name = f"inter_{feat1[:15]}___{feat2[:15]}"
                        long_df[inter_name] = long_df[feat1] * long_df[feat2]
                        interactions.append(inter_name)

            # Add squared terms for top 5
            squares = []
            for feat in top_features[:5]:
                if feat in long_df.columns:
                    sq_name = f"sq_{feat[:20]}"
                    long_df[sq_name] = long_df[feat] ** 2
                    squares.append(sq_name)

            long_feature_cols = long_feature_cols + interactions + squares

        if ADD_GROUP_INTERACTION_FEATURES:

            # Select key features to interact with groups
            key_features = [
                "finance",
                "complexity",
                "governance_composite",
                "planned_expenditure",
                "targets",
                "implementer_performance",
            ]
            key_features = [f for f in key_features if f in long_df.columns]

            # Get group columns (try both grp__ and grp prefix)
            grp_cols = [
                c
                for c in long_df.columns
                if c.startswith("grp__") or c.startswith("grp_")
            ]

            if len(grp_cols) == 0:
                print("No group columns found. Skipping experiment.")
                print(
                    f"Available columns: {[c for c in long_df.columns if 'grp' in c.lower()]}"
                )
                return None, long_df, long_feature_cols

            print(
                f"\nCreating interactions between {len(key_features)} features and {len(grp_cols)} groups"
            )
            print(f"Key features: {key_features}")

            # Create interactions
            interaction_features = []
            for feat in key_features:
                for grp in grp_cols[:10]:  # Limit to top 10 groups to avoid explosion
                    inter_name = f"{feat}__X__{grp}"
                    long_df[inter_name] = long_df[feat] * long_df[grp]
                    interaction_features.append(inter_name)

            long_feature_cols = long_feature_cols + interaction_features

        # train one RF on z
        train_index_long = long_df.index[long_df["activity_id"].isin(train_aids)]

        # ---- debug: sample rows used for training + prediction (features + target) ----
        np.random.seed(0)

        show_cols = (
            ["activity_id", "which_group", "y_raw", "mu_g", "sd_g", "y_z"]
            + feature_cols
            # include group dummies only if add_group_dummies(...) has been called
            + [c for c in long_df.columns if c.startswith("grp")]
        )

        train_rows = long_df.index[long_df["activity_id"].isin(train_aids)]
        test_rows = long_df.index[long_df["activity_id"].isin(test_aids)]

        # Create recency weights: recent 20% gets 3x weight
        cutoff_idx = int(len(train_index_long) * 0.8)
        sample_weights = np.ones(len(train_index_long))
        sample_weights[cutoff_idx:] = 3.0

        yhat_all, rf_model = run_random_forest_median_impute_noclip(
            # yhat_all, rf_model = run_ridge_glm_median_impute_noclip(
            data=long_df,
            feature_cols=long_feature_cols,
            target_col="y_z",
            train_index=train_index_long,
            rf_params=rf_params,
            sample_weight=sample_weights,
            ensemble_with_extratrees=True,
        )
        if GET_FEATURE_IMPORTANCES:
            # ---- save importances for the long-form z model too ----
            imp_zagg = one_sd_shift_importance(
                model=rf_model,
                data=long_df,
                feature_cols=long_feature_cols,
                train_idx=train_index_long,
            )

            # make sure all_imps exists before this block; or create a dedicated dict
            all_imps[("zagg", "y_z")] = imp_zagg

            # ---- FEATURE IMPORTANCE TERMINAL OUTPUT ----
            print("\n" + "=" * 80)
            print("FEATURE IMPORTANCE RANKING (RF on zagg y_z)")
            print("=" * 80)
            print(f"\nTop 30 features by importance_abs_1sd:")
            print(
                "\n"
                + imp_zagg[
                    ["feature", "importance_abs_1sd", "delta_pred_1sd", "sd_train"]
                ]
                .head(30)
                .to_string(index=False)
            )

            print(f"\n\nBottom 10 features (least important):")
            print(
                imp_zagg[
                    ["feature", "importance_abs_1sd", "delta_pred_1sd", "sd_train"]
                ]
                .tail(10)
                .to_string(index=False)
            )

        # ---- SHAP ANALYSIS ----
        if not DONT_MAKE_ANY_SHAP_ANALYSIS_OR_PLOTS:
            print("\n" + "=" * 80)
            print("GENERATING SHAP PLOTS")
            print("=" * 80)

            # Prepare data for SHAP
            X_train = long_df.loc[train_index_long, long_feature_cols].astype(float)
            med = X_train.median(numeric_only=True)
            X_train_imp = X_train.fillna(med)

            test_rows_long = long_df.index[long_df["activity_id"].isin(test_aids)]
            X_test = (
                long_df.loc[test_rows_long, long_feature_cols].astype(float).fillna(med)
            )

            # Sample for SHAP (use subset for speed)
            n_shap_sample = min(500, len(X_test))
            X_shap = X_test.sample(n=n_shap_sample, random_state=42)

            print(f"Computing SHAP values for {len(X_shap)} test samples...")
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_shap)

            # Plot 1: SHAP summary plot (beeswarm)
            print("Generating SHAP summary plot...")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
            plt.title("SHAP Feature Importance (zagg y_z)", fontsize=14)
            plt.tight_layout()
            plt.savefig("shap_summary_zagg.png", dpi=150, bbox_inches="tight")
            print("  Saved: shap_summary_zagg.png")
            plt.show()

            # Plot 2: SHAP bar plot (mean absolute SHAP)
            print("Generating SHAP bar plot...")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_shap, plot_type="bar", show=False, max_display=20
            )
            plt.title("Mean |SHAP| by Feature (zagg y_z)", fontsize=14)
            plt.tight_layout()
            plt.savefig("shap_bar_zagg.png", dpi=150, bbox_inches="tight")
            print("  Saved: shap_bar_zagg.png")
            plt.show()

            print("SHAP analysis complete!\n")

        long_df["pred_z"] = pd.Series(yhat_all, index=long_df.index)

        # ---- Start-year trend correction on ZAGG activity-level predictions ----
        # Compute activity-level means for ALL activities (train + test) respecting
        # the same rating-exclusion logic used in the final evaluation below.
        _RATING_GRP_Z = "rating__rating"
        if SKIP_RATINGS_WHEN_REPORTING:
            _nr_mask_all = long_df["which_group"] != _RATING_GRP_Z
            _zagg_all_pred = (
                long_df.loc[_nr_mask_all].groupby("activity_id")["pred_z"].mean()
            )
            _zagg_all_true = (
                long_df.loc[_nr_mask_all].groupby("activity_id")["y_z"].mean()
            )
        else:
            _zagg_all_pred = long_df.groupby("activity_id")["pred_z"].mean()
            _zagg_all_true = long_df.groupby("activity_id")["y_z"].mean()

        _zagg_corr_df = pd.DataFrame(
            {
                "pred_zagg": _zagg_all_pred,
                "start_date": data["start_date"].reindex(_zagg_all_pred.index),
            }
        )

        apply_start_year_trend_correction(
            data=_zagg_corr_df,
            y=_zagg_all_true,
            train_idx=train_aids,
            pred_col="pred_zagg",
        )

        tmp = long_df.loc[
            test_mask, ["activity_id", "which_group", "y_z", "pred_z"]
        ].copy()

        # Evaluate the long-form z model per outcome/group on the same TEST split
        z_by_outcome = zagg_metrics_per_outcome(
            long_df,
            # outcome_cols=use_cols,          # same set used to build long_df
            outcome_cols=train_outcome_cols,
            meta_by_col=META_BY_COL,  # so we can map outcome_col -> dist__units label
            test_aids=test_aids,  # IMPORTANT: same test_aids used for long_df eval
        )
        # handy dict lookup later
        z_by_outcome_map = z_by_outcome.set_index("outcome_col").to_dict(orient="index")

        # ---- eval on TEST rows (row-level) ----
        test_mask = long_df["activity_id"].isin(test_aids)
        yt = long_df.loc[test_mask, "y_z"].astype(float)
        yp = long_df.loc[test_mask, "pred_z"].astype(float)
        m = np.isfinite(yt) & np.isfinite(yp)

        print("\n=== LONG-FORM Z EVAL (row-level) ===")
        print("rows test:", int(test_mask.sum()), "finite:", int(m.sum()))
        if int(m.sum()) >= 2:
            print("R^2:", float(r2_score(yt[m], yp[m])))
        print(
            "RMSE (z units):",
            float(np.sqrt(np.mean((yp[m] - yt[m]) ** 2))) if int(m.sum()) else np.nan,
        )
        print(
            "MAE  (z units):",
            float(np.mean(np.abs(yp[m] - yt[m]))) if int(m.sum()) else np.nan,
        )
        from numpy import corrcoef

        row_r2 = float(r2_score(yt[m], yp[m])) if int(m.sum()) >= 2 else np.nan
        row_corr = float(corrcoef(yt[m], yp[m])[0, 1]) if int(m.sum()) >= 2 else np.nan

        print("\n=== ROW-LEVEL EXTRA ===")
        print("row-level corr:", row_corr)

        # ---- Extract test-set predictions (start-year-corrected) ----
        _test_aid_set = set(map(str, test_aids))
        _idx_test = _zagg_corr_df.index[_zagg_corr_df.index.isin(_test_aid_set)]
        act_true = _zagg_all_true.reindex(_idx_test)
        act_pred = _zagg_corr_df.loc[_idx_test, "pred_zagg"]

        idx = act_true.index.intersection(act_pred.index)
        act_true = act_true.loc[idx].dropna()
        act_pred = act_pred.loc[act_true.index]

        print("\n=== ZAGG EXACT INPUTS (activity means) ===")
        print("n activities in act_true:", len(act_true))
        print("n activities in act_pred:", len(act_pred))
        print("index equal:", act_true.index.equals(act_pred.index))
        print("sample 10 activity means:")
        tmp = (
            pd.DataFrame({"true_mean_z": act_true, "pred_mean_z": act_pred})
            .sample(min(10, len(act_true)), random_state=0)
            .sort_index()
        )
        print(tmp.to_string())

        # Save zagg predictions for cross-validation analysis
        zagg_df = pd.DataFrame(
            {
                "zagg_true": act_true,
                "zagg_pred": act_pred,
            }
        )
        zagg_df.index.name = "activity_id"

        # ---- PEARSON CORRELATION: zagg (y_z mean) vs rating, all subsets ----
        from scipy.stats import pearsonr as _pearsonr

        # zagg_true for ALL activities (not just test), honoring rating exclusion
        if SKIP_RATINGS_WHEN_REPORTING:
            _zagg_all = (
                long_df.loc[long_df["which_group"] != "rating__rating"]
                .groupby("activity_id")["y_z"]
                .mean()
            )
        else:
            _zagg_all = long_df.groupby("activity_id")["y_z"].mean()

        _rating_all = pd.to_numeric(data["out_rating"], errors="coerce").astype(float)
        _merged_pr = pd.DataFrame({"zagg": _zagg_all, "rating": _rating_all}).dropna()

        print("\n" * 10)
        print("PEARSON CORRELATION FOR ENTIRE DATASET")
        if len(_merged_pr) >= 2:
            _r, _p = _pearsonr(_merged_pr["zagg"].values, _merged_pr["rating"].values)
            print(f"  r = {_r:.4f}  (n={len(_merged_pr)}, p={_p:.4e})")
            _zagg_corr_path = (
                DATA_DIR / "outcomes_model_outputs" / "zagg_rating_pearson.json"
            )
            _zagg_corr_path.parent.mkdir(exist_ok=True)
            import json as _json

            with open(_zagg_corr_path, "w") as _f:
                _json.dump(
                    {"r": float(_r), "p": float(_p), "n": int(len(_merged_pr))}, _f
                )
        print("\n" * 10)

        mm = np.isfinite(act_true.values) & np.isfinite(act_pred.values)

        rng = np.random.RandomState(0)
        shuf = act_pred.sample(frac=1.0, random_state=rng).values

        z_r2 = (
            float(r2_score(act_true.values[mm], act_pred.values[mm]))
            if mm.sum() >= 2
            else np.nan
        )
        z_rmse = (
            float(np.sqrt(np.mean((act_pred.values[mm] - act_true.values[mm]) ** 2)))
            if mm.sum()
            else np.nan
        )
        z_mae = (
            float(np.mean(np.abs(act_pred.values[mm] - act_true.values[mm])))
            if mm.sum()
            else np.nan
        )

        # ---- Bootstrap CIs for R^2, Pairwise, and Spearman (zagg) ----
        z_yt = act_true.values[mm]
        z_yp = act_pred.values[mm]

        if len(z_yt) >= 2:

            # R^2 bootstrap CI (using r2_activity since r2_cell is not meaningful for zagg)
            z_r2_ci = bootstrap_ci(
                z_yt, z_yp, lambda y_t, y_p: r2_score(y_t, y_p), n_bootstrap=1000
            )
            print(
                f"\nZAGG R^2 95% CI: [{z_r2_ci['lower']:.4f}, {z_r2_ci['upper']:.4f}]"
            )

            # Pairwise ordering probability
            z_pairwise = pairwise_ordering_prob(z_yt, z_yp)
            z_pairwise_ci = bootstrap_ci(
                z_yt, z_yp, pairwise_ordering_prob, n_bootstrap=1000
            )
            print(
                f"ZAGG Pairwise ordering: {z_pairwise:.4f}  95% CI: [{z_pairwise_ci['lower']:.4f}, {z_pairwise_ci['upper']:.4f}]"
            )

            # Spearman correlation
            z_spearman = spearman_correlation(z_yt, z_yp)
            z_spearman_ci = bootstrap_ci(
                z_yt, z_yp, spearman_correlation, n_bootstrap=1000
            )
            print(
                f"ZAGG Spearman correlation: {z_spearman:.4f}  95% CI: [{z_spearman_ci['lower']:.4f}, {z_spearman_ci['upper']:.4f}]"
            )

            # Within-group pairwise (grouped by reporting org)
            act_ids_filtered = act_true.index[mm]
            z_orgs = data.loc[act_ids_filtered, "reporting_orgs"].astype(str).values
            _r_zagg = within_group_pairwise_ordering_prob(z_yt, z_yp, z_orgs)
            z_wg_pairwise, z_wg_n_pairs, z_wg_n_groups = (
                _r_zagg["prob"],
                _r_zagg["n_pairs"],
                _r_zagg["n_groups"],
            )
            z_wg_pairwise_ci = bootstrap_ci(
                z_yt,
                z_yp,
                lambda yt, yp: within_group_pairwise_ordering_prob(yt, yp, z_orgs)[
                    "prob"
                ],
                n_bootstrap=1000,
            )
            print(
                f"ZAGG WG-POP (within org): {z_wg_pairwise:.4f}  95% CI: [{z_wg_pairwise_ci['lower']:.4f}, {z_wg_pairwise_ci['upper']:.4f}]  (n_pairs={z_wg_n_pairs}, n_orgs={z_wg_n_groups})"
            )
        else:
            z_r2_ci = {"lower": np.nan, "upper": np.nan}
            z_pairwise = np.nan
            z_pairwise_ci = {"lower": np.nan, "upper": np.nan}
            z_spearman = np.nan
            z_spearman_ci = {"lower": np.nan, "upper": np.nan}
            z_wg_pairwise = np.nan
            z_wg_pairwise_ci = {"lower": np.nan, "upper": np.nan}
            z_wg_n_pairs = 0
            z_wg_n_groups = 0

        scores[("zagg", "outcomes_z_activity_mean")] = {
            "variant": "zagg",
            "outcome_col": "outcomes_z_activity_mean",
            "which_distribution": "ALL_SELECTED_OUTCOMES",
            "which_units": "z",
            "r2_activity": z_r2,
            "r2_cell": np.nan,  # not meaningful here
            "r2_cell_ci_lower": z_r2_ci["lower"],  # using r2_activity CI here
            "r2_cell_ci_upper": z_r2_ci["upper"],
            "pairwise": float(z_pairwise) if np.isfinite(z_pairwise) else np.nan,
            "pairwise_ci_lower": z_pairwise_ci["lower"],
            "pairwise_ci_upper": z_pairwise_ci["upper"],
            "spearman": float(z_spearman) if np.isfinite(z_spearman) else np.nan,
            "spearman_ci_lower": z_spearman_ci["lower"],
            "spearman_ci_upper": z_spearman_ci["upper"],
            "wg_pairwise": (
                float(z_wg_pairwise) if np.isfinite(z_wg_pairwise) else np.nan
            ),
            "wg_pairwise_ci_lower": z_wg_pairwise_ci["lower"],
            "wg_pairwise_ci_upper": z_wg_pairwise_ci["upper"],
            "wg_pairwise_n_pairs": z_wg_n_pairs,
            "wg_pairwise_n_groups": z_wg_n_groups,
            "rmse_cell": z_rmse,  # rmse in z units
            "mae_cell": z_mae,
            "top1_feat": np.nan,  # not meaningful for aggregate
            "top1_abs_1sd": np.nan,
            "n_activities": int(mm.sum()),
            "n_total": int(len(data_raw_univ)),
            "n_train": int(len(train_aids)),
            "n_val": int(len(val_aids)),
            "n_held": int(len(held_aids)),
            "n_test": int(len(test_aids)),
        }

        # ---- SIMPLE MODEL: ZAGG WITH ONLY PLANNED_EXPENDITURE + SECTOR CLUSTERS ----
        print("\n" + "=" * 80)
        print("SIMPLE MODEL: ZAGG with only planned_expenditure + sector clusters")
        print("=" * 80)

        # Identify simple features
        simple_features = ["planned_expenditure"]
        sector_cluster_cols = [
            c for c in long_df.columns if c.startswith("sector_cluster_")
        ]
        simple_features.extend(sector_cluster_cols)

        # Filter to features that exist
        simple_features = [f for f in simple_features if f in long_df.columns]

        print(f"Simple model features: {len(simple_features)}")
        print(
            f"  - planned_expenditure: {'YES' if 'planned_expenditure' in simple_features else 'NO'}"
        )
        print(f"  - sector clusters: {len(sector_cluster_cols)}")

        if len(simple_features) >= 2:
            # Train simple RF model on same data
            X_simple_train = long_df.loc[train_index_long, simple_features].astype(
                float
            )
            y_simple_train = long_df.loc[train_index_long, "y_z"].astype(float)

            # Median imputation
            med_simple = X_simple_train.median(numeric_only=True)
            X_simple_train_imp = X_simple_train.fillna(med_simple)

            # Train simple RF with same params as full model
            from sklearn.ensemble import RandomForestRegressor

            rf_simple = RandomForestRegressor(**rf_params)

            # Apply same recency weights
            rf_simple.fit(
                X_simple_train_imp, y_simple_train, sample_weight=sample_weights
            )

            # Predict on test
            X_simple_test = long_df.loc[test_mask, simple_features].astype(float)
            X_simple_test_imp = X_simple_test.fillna(med_simple)
            pred_simple = rf_simple.predict(X_simple_test_imp)

            # Aggregate to activity level
            long_df_test_simple = long_df.loc[test_mask].copy()
            long_df_test_simple["pred_z_simple"] = pred_simple

            act_pred_simple = long_df_test_simple.groupby("activity_id")[
                "pred_z_simple"
            ].mean()
            act_true_simple = long_df_test_simple.groupby("activity_id")["y_z"].mean()

            # Align indices
            idx_simple = act_true_simple.index.intersection(act_pred_simple.index)
            act_true_simple = act_true_simple.loc[idx_simple]
            act_pred_simple = act_pred_simple.loc[idx_simple]

            # Calculate metrics
            mm_simple = act_true_simple.notna() & act_pred_simple.notna()
            if mm_simple.sum() >= 2:
                r2_simple = float(
                    r2_score(
                        act_true_simple.values[mm_simple],
                        act_pred_simple.values[mm_simple],
                    )
                )
                rmse_simple = float(
                    np.sqrt(
                        np.mean(
                            (
                                act_true_simple.values[mm_simple]
                                - act_pred_simple.values[mm_simple]
                            )
                            ** 2
                        )
                    )
                )

                # Pairwise ordering
                z_pairwise_simp = pairwise_ordering_prob(
                    act_true_simple.values[mm_simple], act_pred_simple.values[mm_simple]
                )
                corr_simple = spearman_correlation(
                    act_true_simple.values[mm_simple], act_pred_simple.values[mm_simple]
                )

                print(f"\nSimple Model Results (Test Set, n={mm_simple.sum()}):")
                print(f"  R^2: {r2_simple:.4f}")
                print(f"  RMSE: {rmse_simple:.4f}")
                print(f"  Spearman: {corr_simple:.4f}")
                print(f"  z_pairwise: {z_pairwise_simp:.4f}")

                print(f"\nFull Model Results (Test Set, for comparison):")
                print(f"  R^2: {z_r2:.4f}")
                print(f"  RMSE: {z_rmse:.4f}")
                print(f"  Spearman: {z_spearman:.4f}")

                print(f"\nDifference (Full - Simple):")
                print(f"  DeltaR^2: {z_r2 - r2_simple:+.4f}")
                print(f"  DeltaRMSE: {z_rmse - rmse_simple:+.4f}")
                print(f"  DeltaSpearman: {z_spearman - corr_simple:+.4f}")

                if abs(z_r2 - r2_simple) < 0.02:
                    print(
                        "\nWARNING: Simple model performs nearly as well as full model!"
                    )
                    print("    Complex feature engineering may not be necessary.")
                elif z_r2 - r2_simple > 0.05:
                    print(
                        "\n Full model shows meaningful improvement over simple model."
                    )
                    print("    Feature engineering provides value.")
            else:
                print("Warning: Not enough data for simple model evaluation")
        else:
            print("Warning: Not enough simple features available")

        print("=" * 80)

        # ---- SAVE ZAGG MODEL OUTPUTS FOR CROSS-MODEL COMPARISON ----
        print("\n" + "=" * 80)
        print("SAVING ZAGG MODEL OUTPUTS FOR COMPARISON ANALYSIS")
        print("=" * 80)

        output_dir_zagg = DATA_DIR / "outcomes_model_outputs"
        output_dir_zagg.mkdir(exist_ok=True)

        # 1. Save feature importances from zagg model
        if ("zagg", "y_z") in all_imps and all_imps[("zagg", "y_z")] is not None:
            imp_zagg_df = all_imps[("zagg", "y_z")].copy()
            imp_zagg_df["variant"] = "zagg"
            imp_zagg_df["outcome_col"] = "outcomes_z_activity_mean"
            # Rename columns to match expected format
            if "importance_abs_1sd" in imp_zagg_df.columns:
                imp_zagg_df = imp_zagg_df.rename(
                    columns={"importance_abs_1sd": "importance"}
                )
            importance_path = output_dir_zagg / "feature_importances.csv"
            imp_zagg_df.to_csv(importance_path, index=False)
            print(f"Saved zagg feature importances to {importance_path}")
            print(f"  {len(imp_zagg_df)} features")
        else:
            print("Warning: No zagg importances found in all_imps")

        # 2. Save zagg predictions
        # Zagg is computed by taking the mean of y_z and pred_z per activity
        predictions_list_zagg = []
        if (
            "long_df" in locals()
            and len(long_df) > 0
            and "y_z" in long_df.columns
            and "pred_z" in long_df.columns
        ):
            # Aggregate by activity_id to get activity-level predictions (zagg)
            zagg_true = long_df.groupby("activity_id")["y_z"].mean()
            zagg_pred = long_df.groupby("activity_id")["pred_z"].mean()

            # Combine and determine split
            for aid in zagg_true.index:
                if aid in zagg_pred.index:
                    split = (
                        "val"
                        if aid in val_aids
                        else ("train" if aid in train_aids else "test")
                    )
                    predictions_list_zagg.append(
                        {
                            "activity_id": aid,
                            "outcome_col": "outcomes_z_activity_mean",
                            "y_true": zagg_true.loc[aid],
                            "y_pred": zagg_pred.loc[aid],
                            "split": split,
                        }
                    )

            if len(predictions_list_zagg) > 0:
                predictions_df = pd.DataFrame(predictions_list_zagg)
                predictions_path = output_dir_zagg / "predictions.csv"
                predictions_df.to_csv(predictions_path, index=False)
                print(f"Saved zagg predictions to {predictions_path}")
                print(f"  {len(predictions_df)} predictions")
                print(
                    f"  Train: {(predictions_df['split']=='train').sum()}, "
                    f"Val: {(predictions_df['split']=='val').sum()}, "
                    f"Test: {(predictions_df['split']=='test').sum()}"
                )

                # Also copy to iati_extractions
                import shutil as _shutil

                _iati_out_dir = (
                    Path.home() / "Code" / "iati_extractions" / "quantitative_outcomes"
                )
                _iati_out_dir.mkdir(parents=True, exist_ok=True)
                _shutil.copy(predictions_path, _iati_out_dir / "zagg_predictions.csv")
                print(
                    f"  Copied zagg predictions to iati_extractions/quantitative_outcomes/"
                )
            else:
                print("Warning: No zagg predictions to save")

            # Save all individual outcome values (long_df) for all activities
            _long_cols = ["activity_id", "which_group", "y_raw"]
            if "y_z" in long_df.columns:
                _long_cols.append("y_z")
            if "pred_z" in long_df.columns:
                _long_cols.append("pred_z")
            _avail = [
                c for c in _long_cols if c in long_df.columns or c == "activity_id"
            ]
            _outcome_vals = long_df.reset_index()[_avail].copy()
            _outcome_vals_path = output_dir_zagg / "all_outcome_values.csv"
            _outcome_vals.to_csv(_outcome_vals_path, index=False)
            print(f"Saved all individual outcome values to {_outcome_vals_path}")
            if "_iati_out_dir" in dir():
                _shutil.copy(
                    _outcome_vals_path, _iati_out_dir / "all_outcome_values.csv"
                )
                print(f"  Copied to iati_extractions/quantitative_outcomes/")
        else:
            print("Warning: long_df not available or missing y_z/pred_z columns")

        print("=" * 80)

        _feat_save_dir = DATA_DIR / "feature_lists"
        _feat_save_dir.mkdir(parents=True, exist_ok=True)
        _feat_save_path = _feat_save_dir / "cost_effectiveness_features.json"
        with open(_feat_save_path, "w") as _f:
            json.dump(
                {
                    "model": "cost_effectiveness",
                    "n_features": len(long_feature_cols),
                    "features": long_feature_cols,
                },
                _f,
                indent=2,
            )
        print(
            f"[feature_lists] Saved {len(long_feature_cols)} features to {_feat_save_path}"
        )

    if not DISABLE_ALL_BUT_ZAGG:
        # for y_outcome_col in outcome_cols:
        for y_outcome_col in train_outcome_cols:
            data_outcome = data[data[y_outcome_col].notna()].copy()
            # for y_outcome_col in ["out_rating"]:
            # data_outcome = data[data[y_outcome_col].notna()]
            X = data_outcome[feature_cols]
            y = data_outcome[y_outcome_col].astype(float)

            if DROP_INCOMPLETE:
                print(
                    "WARNING: DROP_INCOMPLETE=True. Training/val/held will EXCLUDE incomplete activities."
                )

                # right before DROP_INCOMPLETE filtering
                missing_is_completed = data_outcome["is_completed"].isna()
                print("Missing is_completed:", missing_is_completed.sum())
                print(data_outcome.index[missing_is_completed].tolist())

                complete_mask = data_outcome["is_completed"].fillna(0).astype(int) == 1

                # stash incomplete rows (for optional test-only add-back)
                X_incomp = X.loc[~complete_mask]
                y_incomp = y.loc[~complete_mask]
                data_outcome_incomp = data_outcome.loc[~complete_mask]

                # restrict split universe to complete only
                X = X.loc[complete_mask]
                y = y.loc[complete_mask]
                data_outcome = data_outcome.loc[complete_mask]
            else:
                X_incomp = X.iloc[0:0]
                y_incomp = y.iloc[0:0]
                data_outcome_incomp = data_outcome.iloc[0:0]

            print(f"Total samples: {len(X)}")
            print(f"Features: {feature_cols}")

            print("\n=== RATING / PREDICTION DISTRIBUTIONS ===")
            print("\nRaw counts in y:")
            print(y.value_counts().sort_index())
            print(f"Total in y: {len(y)}")

            # REQUIRE these (crash if missing)
            if data_outcome["reporting_orgs"].isna().any():
                bad = int(data_outcome["reporting_orgs"].isna().sum())
                raise ValueError(
                    f"reporting_orgs has {bad} NaNs -- fix upstream (don't silently drop)."
                )

            if data_outcome["start_date"].isna().any():
                bad = int(data_outcome["start_date"].isna().sum())
                raise ValueError(
                    f"start_date has {bad} NaNs -- fix upstream (don't silently drop)."
                )

            train_idx, val_idx, held_idx = split_latest_by_date_with_cutoff(
                data_outcome, date_col="start_date"
            )

            X_train = X.loc[train_idx]
            X_val = X.loc[val_idx]
            X_test = X.loc[held_idx]
            y_train = y.loc[train_idx]
            y_val = y.loc[val_idx]
            y_test = y.loc[held_idx]

            dropped_df = data_outcome_incomp if DROP_INCOMPLETE else None

            if USE_VAL_IN_TRAIN:
                print(
                    "WARNING: USE_VAL_IN_TRAIN=True -- evaluating on held-out TEST set."
                )
                X_train = pd.concat([X_train, X_val], axis=0)
                y_train = pd.concat([y_train, y_val], axis=0)
                X_held = X.loc[held_idx]
                y_held = y.loc[held_idx]
            else:
                print("running on validation set")
                X_test = X_val
                y_test = y_val

            # ---- add incomplete back ONLY to test (inflates test; doesn't affect target sizes) ----
            if DROP_INCOMPLETE and ADD_INCOMPLETE_TO_TEST:
                # only add rows that aren't already in test (shouldn't happen, but safe)
                add_idx = X_incomp.index.difference(X_test.index)

                X_test = pd.concat([X_test, X_incomp.loc[add_idx]], axis=0)
                y_test = pd.concat([y_test, y_incomp.loc[add_idx]], axis=0)

                print(
                    f"NOTE: added {len(add_idx)} incomplete rows to TEST only (test is now {len(X_test)} rows)."
                )

            train_idx = X_train.index
            test_idx = X_test.index

            idx_all = data_outcome.index[data_outcome[y_outcome_col].notna()]
            train_idx_t = pd.Index(train_idx).intersection(idx_all)
            test_idx_t = pd.Index(test_idx).intersection(idx_all)

            rf_params = {
                "n_estimators": 638,  # 1200,      # more trees to reduce variance
                "max_depth": 14,  # 8,
                "min_samples_split": 20,
                "min_samples_leaf": 20,  # 25,
                "max_features": 0.488,  # 0.25,
                "bootstrap": True,
                "max_samples": 0.86,  # 0.75,       # row subsampling (sklearn >= 0.22)
                "ccp_alpha": 1.26e-6,  # 1e-4,
                "random_state": 43,
                "n_jobs": -1,
            }

            trained_outcome_cols, models_by_outcome, runner_used = (
                train_rfs_for_outcomes(
                    data=data_outcome,
                    feature_cols=feature_cols,
                    outcome_cols=[y_outcome_col],
                    train_idx=train_idx,
                    rf_params=rf_params,
                    min_train=30,
                    pred_prefix="pred_",
                    return_models=True,
                )
            )
            if not trained_outcome_cols:
                print(
                    f"SKIP EVAL {y_outcome_col}: no model trained (too few TRAIN labels)."
                )
                continue

            model = models_by_outcome[y_outcome_col]
            imp_df = one_sd_shift_importance(
                model=model,
                data=data_outcome,
                feature_cols=feature_cols,
                train_idx=train_idx,
            )
            # top1 from the full table
            if len(imp_df):
                top1_feat = str(imp_df.iloc[0]["feature"])
                top1_abs_1sd = float(imp_df.iloc[0]["importance_abs_1sd"])
            else:
                top1_feat = None
                top1_abs_1sd = np.nan

            if runner_used[y_outcome_col] == "rf":
                runner = run_random_forest_median_impute_noclip
                runner_kwargs = {
                    "rf_params": rf_params,
                    "ensemble_with_extratrees": True,
                }
            else:
                runner = run_ridge_glm_median_impute_noclip
                runner_kwargs = {}

            pred_col = f"pred_{y_outcome_col}"

            # 5) evaluate once per activity (aggregate across outcome-types)
            per_activity_eval, metrics = evaluate_outcomes_once_per_activity(
                data=data_outcome,
                test_idx=test_idx,
                outcome_cols=trained_outcome_cols,
                pred_prefix="pred_",
                min_outcomes_per_activity=1,
                print_worst=15,
            )
            variant = "dpu" if y_outcome_col.startswith("out_dpu_") else "raw"
            all_imps[(variant, y_outcome_col)] = imp_df
            meta = META_BY_COL.get(
                y_outcome_col, {"which_distribution": "?", "which_units": "?"}
            )

            score_key = (variant, y_outcome_col)  # or f"{variant}::{y_outcome_col}"
            model = models_by_outcome[y_outcome_col]

            # choose same fitting function for permutation
            if runner_used[y_outcome_col] == "rf":
                runner = run_random_forest_median_impute_noclip
                runner_kwargs = {
                    "rf_params": rf_params,
                    "ensemble_with_extratrees": True,
                }
            else:
                runner = run_ridge_glm_median_impute_noclip
                runner_kwargs = {}

            r2_gate = float(metrics["r2_cell"])

            imp = top1_abs_1sd_shift_importance(
                model=model,
                data=data_outcome,
                feature_cols=feature_cols,
                train_idx=train_idx,
                n_eval=200,
                random_state=0,
            )

            # counts for this outcome (non-null y only)
            idx_all = data_outcome.index[data_outcome[y_outcome_col].notna()]
            n_total = int(len(idx_all))

            train_idx_t = pd.Index(train_idx).intersection(idx_all)
            val_idx_t = pd.Index(val_idx).intersection(idx_all)
            held_idx_t = pd.Index(held_idx).intersection(idx_all)

            n_train = int(len(train_idx_t))
            n_val = int(len(val_idx_t))
            n_held = int(len(held_idx_t))
            n_test = int(len(pd.Index(test_idx).intersection(idx_all)))

            scores[score_key] = {
                "variant": variant,
                "outcome_col": y_outcome_col,
                "which_distribution": meta["which_distribution"],
                "which_units": meta["which_units"],
                **metrics,
                # "perm_p_r2": perm["p_value"],
                "top1_feat": top1_feat,  # imp["top1_feat"],
                "top1_abs_1sd": top1_abs_1sd,  # imp["top1_abs_1sd"],
                "n_total": n_total,
                "n_train": n_train,
                "n_val": n_val,
                "n_held": n_held,
                "n_test": n_test,
            }

            zrow = z_by_outcome_map.get(y_outcome_col, {})
            scores[score_key].update(
                {
                    "zmodel_r2": zrow.get("zmodel_r2", np.nan),
                    "zmodel_rmse": zrow.get("zmodel_rmse", np.nan),
                    "zmodel_mae": zrow.get("zmodel_mae", np.nan),
                    "zmodel_n_rows": zrow.get("zmodel_n_rows", np.nan),
                }
            )

            df_scores = pd.DataFrame(list(scores.values()))
            if len(df_scores):
                df_scores = df_scores.sort_values(
                    ["variant", "which_distribution", "which_units", "outcome_col"]
                )

            # Drop r2_cell from display (useless column)
            df_display = df_scores.copy()
            cols_to_drop = [c for c in ["r2_cell"] if c in df_display.columns]
            if cols_to_drop:
                df_display = df_display.drop(columns=cols_to_drop)

            print("\n=== R2 / RMSE summary (all groups) ===")
            print(df_display.to_string(index=False))
            print("")
            print("")

            # Generate LaTeX table
            print_latex_outcome_table(df_scores)
            print("")
            print("")

            print("\n=== 1SD SHIFT IMPORTANCES (per outcome) ===")
            for (variant, ycol), imp_df in all_imps.items():
                print("\n" + "=" * 100)
                print(f"{variant}  {ycol}")
                if imp_df is None or len(imp_df) == 0:
                    print("(no importances)")
                    continue
                print(
                    imp_df[
                        ["feature", "importance_abs_1sd", "delta_pred_1sd", "sd_train"]
                    ].to_string(index=False, float_format=lambda x: f"{x: .6g}")
                )

            print("")
            print("")

    # ---- FINAL BOOTSTRAP SUMMARY: ZAGG (RF + LLM Forecast + recency) ----
    if ("zagg", "outcomes_z_activity_mean") in scores:
        print("\n" + "=" * 80)
        print("BOOTSTRAP 95% CI: ZAGG (Validation Set)")
        print("=" * 80)

        zagg_scores = scores[("zagg", "outcomes_z_activity_mean")]
        r2_val = zagg_scores.get("r2_activity", np.nan)
        r2_lower = zagg_scores.get("r2_cell_ci_lower", np.nan)
        r2_upper = zagg_scores.get("r2_cell_ci_upper", np.nan)

        pairwise_val = zagg_scores.get("pairwise", np.nan)
        pairwise_lower = zagg_scores.get("pairwise_ci_lower", np.nan)
        pairwise_upper = zagg_scores.get("pairwise_ci_upper", np.nan)

        n_val = zagg_scores.get("n_val", 0)

        wg_pairwise_val = zagg_scores.get("wg_pairwise", np.nan)
        wg_pairwise_lower = zagg_scores.get("wg_pairwise_ci_lower", np.nan)
        wg_pairwise_upper = zagg_scores.get("wg_pairwise_ci_upper", np.nan)
        wg_n_pairs = zagg_scores.get("wg_pairwise_n_pairs", 0)
        wg_n_groups = zagg_scores.get("wg_pairwise_n_groups", 0)

        print(
            f"R^2 = {r2_val:.4f}  [95% CI: {r2_lower:.4f}, {r2_upper:.4f}]  (n={n_val})"
        )
        print(
            f"Pairwise = {pairwise_val:.4f}  [95% CI: {pairwise_lower:.4f}, {pairwise_upper:.4f}]"
        )
        print(
            f"WG-POP (within org) = {wg_pairwise_val:.4f}  [95% CI: {wg_pairwise_lower:.4f}, {wg_pairwise_upper:.4f}]  (n_pairs={wg_n_pairs}, n_orgs={wg_n_groups})"
        )
        print("=" * 80)

    # ---- SAVE METADATA FOR CROSS-MODEL COMPARISON ----
    # Note: Feature importances and predictions are saved inside the zagg block above
    # This section only saves metadata which is accessible via the scores dict

    output_dir = DATA_DIR / "outcomes_model_outputs"
    output_dir.mkdir(exist_ok=True)

    # Save metadata
    metadata = {
        "model_name": "Outcomes (Cost-effectiveness)",
        "target_description": "Quantitative outcomes (benefit-cost ratios, yields, emissions, etc.) and zagg aggregate",
        "n_features": len(feature_cols) if "feature_cols" in locals() else None,
        "feature_names": feature_cols if "feature_cols" in locals() else [],
        "n_outcomes": len(train_outcome_cols),
        "outcome_names": train_outcome_cols,
        "n_train": scores.get(("zagg", "outcomes_z_activity_mean"), {}).get(
            "n_train", None
        ),
        "n_val": scores.get(("zagg", "outcomes_z_activity_mean"), {}).get(
            "n_val", None
        ),
        "n_test": scores.get(("zagg", "outcomes_z_activity_mean"), {}).get(
            "n_test", None
        ),
        "val_r2": scores.get(("zagg", "outcomes_z_activity_mean"), {}).get(
            "r2_activity", None
        ),
        "test_r2": scores.get(("zagg", "outcomes_z_activity_mean"), {}).get(
            "r2_activity", None
        ),  # using val as test
        "timestamp": datetime.now().isoformat(),
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    # Print train/val/test counts for all outcome categories
    print("\n" + "=" * 80)
    print("=== OUTCOME CATEGORY COUNTS (train / val / test) ===")
    print("=" * 80)
    try:
        count_cols = [
            "variant",
            "outcome_col",
            "n_train",
            "n_val",
            "n_held",
            "n_test",
            "n_total",
        ]
        avail = [c for c in count_cols if c in df_scores.columns]
        outcome_rows = df_scores[df_scores["outcome_col"] != "outcomes_z_activity_mean"]
        print(outcome_rows[avail].to_string(index=False))
    except NameError:
        print("(df_scores not available)")
    print("=" * 80)

    print(f"\nAll outputs saved to {output_dir}")
    print("=" * 80)

    print("")
    print("")


if __name__ == "__main__":
    main()
