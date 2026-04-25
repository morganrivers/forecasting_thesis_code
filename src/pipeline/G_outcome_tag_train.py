"""
Track 1: Probabilistic forecasting of outcome tags.

This is probabilistic forecasting -- we care about the full probability, not a binary
threshold.

Uses the same features as A_overall_rating_fit_and_evaluate.py (predictors available at activity start).

Splits:
  train: activities with start_date <= LATEST_TRAIN_POINT
  val:   LATEST_TRAIN_POINT < start_date <= LATEST_VALIDATION_POINT
  test:  LATEST_VALIDATION_POINT < start_date < TOO_LATE_CUTOFF

Output: data/outcome_tags/tag_model_results.json + data/outcome_tags/tag_predictions.csv

"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=UserWarning)


ADD_RATINGS = False  # If True, include overall evaluation rating as a feature

ADD_START_YEAR_CORRECTION = True  # fit residual ~ start_year on train, apply to all
# If True (original behaviour): correction applied to RF probas BEFORE averaging with ET,
# so the net effect on the ensemble is correction/2.
# If False: correction applied to the averaged RF+ET ensemble (full correction magnitude).
CORRECT_RF_BEFORE_ET = True
SKIP_START_YEAR_CORRECTION_TAGS = {
    "tag_monitoring_and_evaluation_challenges",
    "tag_high_disbursement",
}
ADD_LLM_CORRECTION = False  # removed: LLM averaging doesn't generalise to kfold


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
UTILS_DIR = SRC_DIR / "utils"
DATA_DIR = SRC_DIR.parent / "data"
OUT_DIR = DATA_DIR / "outcome_tags"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    pick_start_date,
    restrict_to_reporting_orgs_exact,
)
from ml_models import DISABLE_ET
from scoring_metrics import (
    brier_skill_score,
    side_accuracy,
)
from scoring_metrics import (
    pairwise_ordering_prob_excl_ties as pairwise_ordering_prob,
)
from scoring_metrics import (
    within_group_pairwise_ordering_prob as _wg_pop_fn,
)
from sklearn.metrics import average_precision_score, roc_auc_score

# ---- Paths (match C_run_GLM_nobayes.py) ----
INFO_FOR_ACTIVITY_FORECASTING = (
    DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"
)
MERGED_OVERALL_RATINGS = DATA_DIR / "merged_overall_ratings.jsonl"
TARGETS_CONTEXT_MAPS = DATA_DIR / "outputs_targets_context_maps.jsonl"
FINANCE_SECTORS = (
    DATA_DIR / "outputs_finance_sectors_disbursements_baseline_gemini2p5flash.jsonl"
)
ALL_GRADES = str(DATA_DIR / "*_grades.jsonl")
LLM_EXPENDITURE_JSONL = DATA_DIR / "llm_planned_expenditure.jsonl"
APPLIED_TAGS = OUT_DIR / "applied_tags.jsonl"
LLM_FORECAST_PROBS = OUT_DIR / "llm_tag_forecast_probs.csv"
OUT_RESULTS = OUT_DIR / "tag_model_results.json"
OUT_PREDICTIONS = OUT_DIR / "tag_predictions.csv"
OUT_MODELS = OUT_DIR / "tag_models.pkl"
OUT_REGULARIZATION = OUT_DIR / "tag_regularization_results.json"  # written by F script
OUT_YEAR_CORR = OUT_DIR / "tag_year_correction_data.pkl"

# ---- Split config (same as GLM) ----
from leakage_risk import EXCLUDE_TEST_LEAKAGE_RISK, TEST_LEAKAGE_RISK_IDS
from split_constants import (
    LATEST_TRAIN_POINT,
    LATEST_VALIDATION_POINT,
    TOO_LATE_CUTOFF,
    split_latest_by_date_with_cutoff,
)

# ---- Model config ----
ACC_SKILL_MIN_IMPROVEMENT = (
    0.005  # model val_acc must beat majority baseline by at least this
)
MIN_TAG_TRAIN_COUNT = 150  # Track 1: min positives in TRAIN for reliable proba model
MIN_TAG_TRAIN_COUNT_LOW = (
    30  # also train models for tags with 30-149 positives (lower confidence)
)
MIN_TAG_VAL_COUNT = 5  # min positives in VAL to evaluate
NUM_ORGS_KEEP = 4
VERBOSE = True  # Print detailed data / feature / tag stats before training
EXCLUDE_ATTEMPTED_TAGS = True  # If True, skip all tag columns ending in _attempted

# ---- Thesis invariant: class weight threshold ----
# Thesis RF_ET_ensemble: above 65% positive -> class_weight=None; at or below -> "balanced"
CLASS_WEIGHT_POS_RATE_THRESHOLD = 0.65

# ---- Thesis invariant: start-year ridge penalty ----
# Thesis start_year_trend_correction: alpha=50 shrinks the year slope toward zero
START_YEAR_RIDGE_ALPHA = 50.0

RF_N_ESTIMATORS = 500  # overridden at runtime by --save_n_estimators
# Single RF param set matching D_train_staged's original training (min_samples_leaf=5,
# no max_depth).  class_weight is overridden per-tag inside train_rf_et_ensemble.
RF_PARAMS_BASE = {
    "n_estimators": 500,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "oob_score": True,
}
# Aliases kept for any external imports that reference these names
RF_PARAMS_RICH = RF_PARAMS_BASE
RF_PARAMS_SHALLOW = RF_PARAMS_BASE
ADAPTIVE_DEPTH_THRESHOLD = 0  # threshold never triggered: always uses RF_PARAMS_BASE

# Per-tag RF param overrides (merged on top of RF_PARAMS_BASE before training).
# Omitted keys inherit from the base.  max_depth=None means unlimited (default).
TAG_RF_PARAMS_OVERRIDES: dict[str, dict] = {
    "tag_high_disbursement": {"min_samples_leaf": 20},
    "tag_policy_regulatory_reforms_success_success": {"min_samples_leaf": 20},
    "tag_over_budget_success": {"min_samples_leaf": 40},
    "tag_capacity_building_delivered_success": {"min_samples_leaf": 20},
    "tag_high_beneficiary_satisfaction_or_reach_success": {"max_depth": 10},
    "tag_targets_met_or_exceeded_success": {"max_depth": 10},
    "tag_private_sector_engagement_success": {"min_samples_leaf": 20},
    "tag_monitoring_and_evaluation_challenges": {"max_depth": 10},
    "tag_gender_equitable_outcomes_success": {"min_samples_leaf": 20},
    "tag_funds_reallocated": {"min_samples_leaf": 10},
}

# Tags that bypass per-tag feature selection -- always trained on all features.
TAGS_SKIP_FEATURE_SELECTION: set[str] = {
    "tag_high_disbursement",
}

# Tags that always keep their RF+ET model -- the const_base quality gate is skipped.
TAGS_SKIP_CONST_BASE_GATE: set[str] = {
    "tag_high_disbursement",
}

# Hardcoded set of 14 tags used when --hardcode_tags is passed (bypasses val-run JSON).
HARDCODED_14_TAGS: list[str] = [
    "tag_closing_date_extended",
    "tag_funds_cancelled_or_unutilized",
    "tag_funds_reallocated",
    "tag_high_disbursement",
    "tag_improved_financial_performance",
    "tag_project_restructured",
    "tag_targets_revised",
    "tag_capacity_building_delivered_success",
    "tag_gender_equitable_outcomes_success",
    "tag_high_beneficiary_satisfaction_or_reach_success",
    "tag_over_budget_success",
    "tag_policy_regulatory_reforms_success_success",
    "tag_private_sector_engagement_success",
    "tag_targets_met_or_exceeded_success",
]

# ET params override: when set (by --param_suffix), train_rf_et_ensemble uses this
# dict for ExtraTreesClassifier instead of deriving from rf_params.
# class_weight and max_samples are always stripped/overridden by the function.
ET_PARAMS_OVERRIDE: dict | None = None

# ---- Param sets for --param_suffix sweep (Z_sweep_rf_params.py) ----
# Each entry: (rf_params_dict, et_params_dict).  class_weight present but
# overridden per-tag inside train_rf_et_ensemble -- kept here for documentation only.
PARAM_SETS: dict[str, tuple[dict, dict]] = {
    "1": (  # LEAF=3 -- slightly more permissive than winner
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 3,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 3,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "2": (  # LEAF=1 -- fully unpruned, classical RF/ET default
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "3": (  # MORE TREES -- stability check near optimum
        {
            "n_estimators": 800,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 800,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "4": (  # SLIGHTLY WIDER FEATURES -- max_features=0.2 (~11 from 55)
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": 0.2,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": 0.2,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "5": (  # SLIGHTLY NARROWER FEATURES -- log2 (~6 from 55)
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "log2",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "log2",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "6": (  # LIGHT ROW SUBSAMPLING -- max_samples=0.9
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "max_samples": 0.9,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "7": (  # LEAF=3 + WIDER FEATURES -- combine two permissive nudges
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 3,
            "min_samples_split": 2,
            "max_features": 0.2,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 3,
            "min_samples_split": 2,
            "max_features": 0.2,
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "8": (  # ET MORE RANDOM -- log2 on ET, sqrt on RF
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "log2",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "9": (  # ET MORE CAPACITY -- leaf=1 on ET, leaf=5 on RF
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
    "10": (  # ET MORE TREES -- 800 on ET, 500 on RF
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
        {
            "n_estimators": 800,
            "max_depth": None,
            "min_samples_leaf": 5,
            "min_samples_split": 2,
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": 42,
            "n_jobs": -1,
        },
    ),
}

# ---- Per-tag regularization strategy (from F_regularization_strategies.py) ----
# When True, uses top-5 features (by RF importance) for any tag where top5 val
# brier_skill > baseline val brier_skill in the F regularization results.
# Otherwise keeps all features. Feature importances come from the previous D run's
# tag_models.pkl. Falls back to baseline if F results or importances are unavailable.
USE_PER_TAG_STRATEGY = True


# When True: drops governance + uncertainty_flags feature groups from the
# global feature set before training any tag model.
DROP_NOISY_FEATURE_GROUPS = True

NOISY_FEATURE_GROUPS = [
    # governance (7) -- WGI indicators + CPIA + interaction term
    "wgi_control_of_corruption_est",
    "wgi_government_effectiveness_est",
    "wgi_political_stability_est",
    "wgi_regulatory_quality_est",
    "wgi_rule_of_law_est",
    "cpia_score",
    "governance_x_complexity",
    # uncertainty_flags (11) -- missingness indicators
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

# ---- Manual factor blending (post-processing after per-tag RF+ET training) ----
# For tags in TAGS_FACTOR_BLEND, predicted probabilities are blended with a
# manual-factor-based prediction using a linear ramp on minority class count.
# d_weight = clip((minority - BLEND_LO) / (BLEND_HI - BLEND_LO), 0, 1)
# At minority >= BLEND_HI: D_train_staged only.  At minority <= BLEND_LO: full factor.
BLEND_HI = 400
BLEND_LO = 100

# ---- Thesis invariants (checked at import time) ----
assert (
    len(HARDCODED_14_TAGS) == 14
), "thesis outcome_tag_forecasting: exactly 14 forecasted outcome tags"
assert (
    len(set(HARDCODED_14_TAGS)) == 14
), "HARDCODED_14_TAGS contains duplicate tag names"
assert (
    BLEND_LO == 100
), "thesis tag_group_avg: blend ramp starts at 100 minority-class training examples"
assert (
    BLEND_HI == 400
), "thesis tag_group_avg: blend ramp ends at 400 minority-class training examples"
assert BLEND_LO < BLEND_HI, "BLEND_LO must be strictly less than BLEND_HI"
assert (
    CLASS_WEIGHT_POS_RATE_THRESHOLD == 0.65
), "thesis RF_ET_ensemble: class weight threshold is 65%"
assert (
    START_YEAR_RIDGE_ALPHA == 50.0
), "thesis start_year_trend_correction: ridge penalty alpha=50"
# Governance = first 7 entries (comment in NOISY_FEATURE_GROUPS marks the boundary)
_noisy_gov = NOISY_FEATURE_GROUPS[:7]
_noisy_miss = NOISY_FEATURE_GROUPS[7:]
assert (
    len(_noisy_gov) == 7
), f"thesis feature_selection: expected 7 governance features, got {len(_noisy_gov)}"
assert (
    len(_noisy_miss) == 11
), f"thesis feature_selection: expected 11 missingness features, got {len(_noisy_miss)}"
assert (
    RF_PARAMS_BASE["min_samples_leaf"] == 5
), "thesis RF_ET_ensemble: base config min_samples_leaf=5"
assert (
    RF_PARAMS_BASE["max_features"] == "sqrt"
), "thesis RF_ET_ensemble: base config max_features='sqrt'"
assert (
    RF_PARAMS_BASE["n_estimators"] == 500
), "thesis RF_ET_ensemble: base config n_estimators=500"
assert (
    RF_PARAMS_BASE.get("max_depth") is None
), "thesis RF_ET_ensemble: base config max_depth=None (unlimited)"

# Tags eligible for factor blending (rescoping group disbanded -- uses D_train_staged directly)
TAGS_FACTOR_BLEND: list[str] = [
    "tag_high_disbursement",
    "tag_targets_met_or_exceeded_success",
    "tag_high_beneficiary_satisfaction_or_reach_success",
    "tag_private_sector_engagement_success",
    "tag_capacity_building_delivered_success",
    "tag_gender_equitable_outcomes_success",
    "tag_policy_regulatory_reforms_success_success",
    "tag_improved_financial_performance",
    "tag_over_budget_success",
]

# factor_name -> (positive_tags, negative_tags)
# factor_rescoping is kept here for factor computation (its signal helps other tags via OLS)
# even though those tags are not in TAGS_FACTOR_BLEND and are not blended.
MANUAL_FACTORS: dict[str, tuple[list[str], list[str]]] = {
    "factor_success": (
        [
            "tag_targets_met_or_exceeded_success",
            "tag_high_beneficiary_satisfaction_or_reach_success",
            "tag_private_sector_engagement_success",
            "tag_capacity_building_delivered_success",
            "tag_policy_regulatory_reforms_success_success",
            "tag_improved_financial_performance",
        ],
        [],
    ),
    "factor_rescoping": (
        [
            "tag_project_restructured",
            "tag_targets_revised",
            "tag_closing_date_extended",
            "tag_funds_reallocated",
        ],
        [],
    ),
    "factor_finance": (
        ["tag_high_disbursement"],
        ["tag_funds_cancelled_or_unutilized"],
    ),
}


def _compute_manual_factor_scores(tag_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute manual factor scores from observed binary tag values."""
    scores = {}
    for factor_name, (pos_tags, neg_tags) in MANUAL_FACTORS.items():
        pos = [t for t in pos_tags if t in tag_matrix.columns]
        neg = [t for t in neg_tags if t in tag_matrix.columns]
        pos_mean = tag_matrix[pos].mean(axis=1) if pos else 0.0
        neg_mean = tag_matrix[neg].mean(axis=1) if neg else 0.0
        scores[factor_name] = pos_mean - neg_mean
    return pd.DataFrame(scores, index=tag_matrix.index)


def _reconstruct_tag_probas_from_factors(
    factor_preds: pd.DataFrame,
    factor_scores_train: pd.DataFrame,
    tag_matrix_train: pd.DataFrame,
    all_index: pd.Index,
) -> pd.DataFrame:
    """OLS map: factor_scores_train -> tag_values_train, applied to factor_preds."""
    from numpy.linalg import lstsq

    F_tr = factor_scores_train.values
    F_tr_b = np.hstack([F_tr, np.ones((len(F_tr), 1))])
    tag_probas = {}
    for tag in TAGS_FACTOR_BLEND:
        if tag not in tag_matrix_train.columns:
            continue
        y_tr = tag_matrix_train[tag].values
        w, _, _, _ = lstsq(F_tr_b, y_tr, rcond=None)
        F_pred_b = np.hstack(
            [factor_preds.loc[all_index].values, np.ones((len(all_index), 1))]
        )
        tag_probas[tag] = np.clip(F_pred_b @ w, 0.0, 1.0)
    return pd.DataFrame(tag_probas, index=all_index)


def apply_manual_factor_blend(
    data: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    feature_cols: list[str],
    train_medians: pd.Series,
    all_results: list[dict],
    all_probas: dict,
) -> None:
    """
    Post-process D_train_staged results by blending with manual factor predictions.

    For each tag in TAGS_FACTOR_BLEND that isn't const_base:
      1. Train RF+ET regressors on manual factor scores (derived from observed train tags).
      2. Predict factor scores for all activities.
      3. Reconstruct per-tag probabilities via OLS factor->tag mapping.
      4. Blend: d_weight * D_proba + (1-d_weight) * factor_proba, where
         d_weight = clip((minority - BLEND_LO) / (BLEND_HI - BLEND_LO), 0, 1).
      5. Re-evaluate on val and update all_results and all_probas in-place.
    """
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    available_tags = [t for t in TAGS_FACTOR_BLEND if t in data.columns]
    if not available_tags:
        return

    # All tags needed for factor score computation (blend tags + any tags used in factor definitions)
    factor_input_tags = list(
        dict.fromkeys(t for tags, negs in MANUAL_FACTORS.values() for t in tags + negs)
    )
    all_factor_tags = list(
        dict.fromkeys(
            available_tags + [t for t in factor_input_tags if t in data.columns]
        )
    )

    # Build tag matrix (train only, NaN filled with train mean)
    tag_matrix_train = data.loc[train_idx, all_factor_tags].copy()
    for t in all_factor_tags:
        tag_matrix_train[t] = tag_matrix_train[t].fillna(tag_matrix_train[t].mean())

    tag_matrix_all = data[all_factor_tags].copy()
    for t in all_factor_tags:
        tag_matrix_all[t] = tag_matrix_all[t].fillna(tag_matrix_train[t].mean())

    # Factor scores
    factor_scores_all = _compute_manual_factor_scores(tag_matrix_all)
    factor_scores_train = factor_scores_all.loc[train_idx]

    # Feature matrix
    X_all = data[feature_cols].fillna(train_medians)
    X_train = X_all.loc[train_idx].values

    # Train one RF+ET regressor per factor
    _reg_params = dict(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=25,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    factor_preds: dict[str, np.ndarray] = {}
    print("\n[factor_blend] Training factor regressors...")
    for fn in MANUAL_FACTORS:
        if fn not in factor_scores_train.columns:
            continue
        y_tr = factor_scores_train[fn].values
        rf = RandomForestRegressor(**_reg_params)
        rf.fit(X_train, y_tr)
        if DISABLE_ET:
            preds = rf.predict(X_all.values)
            p_val_f = rf.predict(X_all.loc[val_idx].values)
        else:
            et = ExtraTreesRegressor(**{**_reg_params, "bootstrap": False})
            et.fit(X_train, y_tr)
            preds = (rf.predict(X_all.values) + et.predict(X_all.values)) / 2
            p_val_f = (
                rf.predict(X_all.loc[val_idx].values)
                + et.predict(X_all.loc[val_idx].values)
            ) / 2
        factor_preds[fn] = preds
        # Val R^2
        y_val_f = factor_scores_all.loc[val_idx, fn].values
        ss_res = np.sum((y_val_f - p_val_f) ** 2)
        ss_tot = np.sum((y_val_f - y_val_f.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  {fn:30s}  val R^2={r2:+.3f}")

    factor_preds_df = pd.DataFrame(factor_preds, index=data.index)

    # Reconstruct per-tag probabilities from factor predictions
    tag_proba_df = _reconstruct_tag_probas_from_factors(
        factor_preds_df, factor_scores_train, tag_matrix_train, data.index
    )

    # Index results by tag for fast lookup
    results_by_tag: dict[str, int] = {r["tag"]: i for i, r in enumerate(all_results)}

    print("[factor_blend] Blending tag predictions...")
    for tag in available_tags:
        if tag not in tag_proba_df.columns:
            continue
        if tag not in results_by_tag:
            continue
        i = results_by_tag[tag]
        res = all_results[i]
        if res["model_type"] == "const_base":
            continue

        n_train_pos = res.get("train_n_pos", 0)
        n_tr = res.get("train_n", 0)
        minority = min(n_train_pos, n_tr - n_train_pos)
        d_weight = float(
            np.clip((minority - BLEND_LO) / (BLEND_HI - BLEND_LO), 0.0, 1.0)
        )
        assert (
            0.0 <= d_weight <= 1.0
        ), f"thesis tag_group_avg: blend weight must be in [0,1], got {d_weight}"

        if d_weight >= 1.0:
            continue  # no blending needed

        mtype = res["model_type"]
        col_sfx = "rf" if mtype in ("rf", "rf+ET") else mtype
        d_col = f"{tag}__{col_sfx}"
        if d_col not in all_probas:
            continue

        blended_all = (
            d_weight * all_probas[d_col] + (1.0 - d_weight) * tag_proba_df[tag]
        )
        all_probas[d_col] = blended_all

        # Re-evaluate on val
        y_val_raw = data.loc[val_idx, tag].dropna()
        if len(y_val_raw) == 0:
            continue
        base_rate = n_train_pos / n_tr if n_tr > 0 else 0.5
        val_p = blended_all.reindex(y_val_raw.index).fillna(base_rate).values
        new_metrics = _eval_ensemble_probas(
            val_p, y_val_raw.to_numpy().astype(float), tag, mtype, n_train_pos, n_tr
        )
        # Preserve track and train metrics from original result
        for k in (
            "track",
            "train_auc",
            "train_brier",
            "train_brier_skill",
            "train_pairwise_ordering_prob",
            "train_acc",
            "train_logloss",
            "tag_strategy",
            "calibrated",
        ):
            if k in res:
                new_metrics[k] = res[k]
        all_results[i] = new_metrics
        print(
            f"  {tag:52s}  w={d_weight:.2f}  "
            f"POP={new_metrics.get('val_pairwise_ordering_prob', float('nan')):.3f}  "
            f"BrierSkl={new_metrics.get('val_brier_skill', float('nan')):+.3f}"
        )


KEEP_REPORTING_ORGS = [
    "UK - Foreign, Commonwealth Development Office (FCDO)",
    "Asian Development Bank",
    "World Bank",
    "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); Federal Ministry for Economic Cooperation and Development (BMZ)",
]


def get_train_activity_ids_from_dates(
    info_csv: str,
    cutoff_date: str | None = None,
    keep_orgs: list[str] | None = None,
) -> set:
    """Return set of activity_ids whose start_date <= cutoff_date (default: LATEST_TRAIN_POINT).

    keep_orgs: if provided, restrict to activities from these reporting organisations.
               Should be KEEP_REPORTING_ORGS to match the modelled dataset exactly.
    """
    usecols = [
        "activity_id",
        "txn_first_date",
        "actual_start_date",
        "original_planned_start_date",
    ]
    if keep_orgs is not None:
        usecols.append("reporting_orgs")
    df = pd.read_csv(info_csv, usecols=usecols, dtype={"activity_id": str})
    for col in ["txn_first_date", "actual_start_date", "original_planned_start_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["start_date"] = df.apply(pick_start_date, axis=1)
    cutoff = pd.Timestamp(
        cutoff_date if cutoff_date is not None else LATEST_TRAIN_POINT
    )
    mask = df["start_date"].isna() | (df["start_date"] <= cutoff)
    if keep_orgs is not None:
        mask &= df["reporting_orgs"].isin(keep_orgs)
    return set(df.loc[mask, "activity_id"].astype(str))


def load_applied_tags(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load applied_tags.jsonl (new format with unsigned_tags + signed_tags).

    Unsigned tags -> one binary column:  tag_{name}          (0/1)
    Signed tags  -> two binary columns:  tag_{name}_attempted (0/1)
                                         tag_{name}_success   (0/1)

    For signed tags:
      success        -> attempted=1, success=1
      failure        -> attempted=1, success=0
      not_applicable -> attempted=0, success=0

    Returns (DataFrame indexed by activity_id, list of model column names).
    """
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if "activity_id" in obj and (
                "unsigned_tags" in obj or "signed_tags" in obj
            ):
                records.append(obj)

    if not records:
        raise ValueError(f"No records found in {path}")

    # Discover tag names from first record
    sample = records[0]
    unsigned_names = sorted(sample.get("unsigned_tags", {}).keys())
    signed_names = sorted(sample.get("signed_tags", {}).keys())

    # Build model column names
    model_cols = []
    for tag in unsigned_names:
        model_cols.append(f"tag_{tag}")
    for tag in signed_names:
        model_cols.append(f"tag_{tag}_attempted")
        model_cols.append(f"tag_{tag}_success")

    print(f"  {len(unsigned_names)} unsigned tags -> {len(unsigned_names)} columns")
    print(
        f"  {len(signed_names)} signed tags -> {len(signed_names) * 2} columns (attempted + success)"
    )
    print(f"  {len(model_cols)} total model columns across {len(records)} activities")

    rows = []
    for rec in records:
        row = {"activity_id": rec["activity_id"]}

        # Unsigned: simple bool -> 0/1
        for tag in unsigned_names:
            val = rec.get("unsigned_tags", {}).get(tag)
            row[f"tag_{tag}"] = 1 if val else 0

        # Signed: ternary -> two columns
        for tag in signed_names:
            val = rec.get("signed_tags", {}).get(tag)
            if val is True:
                row[f"tag_{tag}_attempted"] = 1
                row[f"tag_{tag}_success"] = 1
            elif val is False:
                row[f"tag_{tag}_attempted"] = 1
                row[f"tag_{tag}_success"] = 0
            else:
                # None / null / missing -> not applicable; NaN excludes row from training
                row[f"tag_{tag}_attempted"] = np.nan
                row[f"tag_{tag}_success"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows).set_index("activity_id")
    return df, model_cols


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


def build_feature_matrix(train_activity_ids=None) -> pd.DataFrame:
    """
    Build the feature matrix replicating C_run_GLM_nobayes.py's join pattern.
    Returns a DataFrame indexed by activity_id with all predictors.
    """
    info_csv = str(INFO_FOR_ACTIVITY_FORECASTING)

    print("Loading grades data (LLM grades: finance, targets, etc.)...")
    grades_df = load_grades(ALL_GRADES)

    print("Loading activity scope...")
    scope_df = load_activity_scope(info_csv)

    print("Loading implementing org type...")
    impl_type_df = load_implementing_org_type(info_csv)

    print("Loading gdp_percap...")
    gdp_df = load_gdp_percap(info_csv)

    print("Loading planned expenditure (LLM-extracted, verified overrides applied)...")
    expend_df = load_llm_planned_expenditure()

    print("Loading world bank indicators...")
    world_bank_df = load_world_bank_indicators(info_csv)

    print("Loading regions...")
    regions_df = pd.read_csv(
        info_csv,
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
        dtype={"activity_id": str},
    ).set_index("activity_id")

    print("Loading is_completed...")
    is_completed = load_is_completed(info_csv)

    print("Loading loan_or_disbursement...")
    lod_df = load_loan_or_disbursement()

    print("Loading target ratings...")
    ratings = load_ratings(str(MERGED_OVERALL_RATINGS))

    print("Loading targets/context maps features...")
    tc_maps_df = load_targets_context_maps_features(TARGETS_CONTEXT_MAPS)

    print("Loading sector cluster allocations...")
    sector_clusters_df = data_sector_clusters(
        str(FINANCE_SECTORS), train_activity_ids=train_activity_ids
    )

    # Start from ratings (same as GLM)
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

    # Encode activity_scope as numeric
    data["activity_scope"] = pd.to_numeric(data["activity_scope"], errors="coerce")

    # Fill NaN region values with 0
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

    # Similarity features (adds rep_org_* cols + knn similarity features)
    data, sim_feature_cols = add_similarity_features(
        data, info_csv, KEEP_REPORTING_ORGS
    )

    # Restrict to 4 orgs (same as GLM)
    data = restrict_to_reporting_orgs_exact(data, KEEP_REPORTING_ORGS)
    print(f"After org restrict: {len(data)} rows")

    # Add dates
    data = add_dates_to_dataframe(data, info_csv)
    llm_dur_df = load_llm_planned_duration()
    data["planned_duration"] = llm_dur_df["planned_duration"].reindex(data.index)

    # Enhanced uncertainty features
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

    # Feature engineering (matching C_run_GLM_nobayes.py)
    # planned_expenditure is ln(USD) as loaded; expose raw USD + keep log separately
    if "planned_expenditure" in data.columns:
        data["log_planned_expenditure"] = data["planned_expenditure"]
        data["planned_expenditure"] = np.exp(data["planned_expenditure"])

    # expenditure per year (raw USD / planned_duration years), NaN-guarded
    if "planned_expenditure" in data.columns and "planned_duration" in data.columns:
        ratio = data["planned_expenditure"] / data["planned_duration"]
        valid = (data["planned_duration"] >= 1) & (
            data["planned_expenditure"] >= 100_000
        )
        data["expenditure_per_year"] = ratio.where(valid, np.nan)
        data["expenditure_per_year_log"] = np.log(ratio.where(valid, np.nan))

    # governance x complexity interaction
    if "complexity" in data.columns:
        wgi_cols = [
            "wgi_control_of_corruption_est",
            "wgi_political_stability_est",
            "wgi_government_effectiveness_est",
            "wgi_regulatory_quality_est",
            "wgi_rule_of_law_est",
        ]
        present = [c for c in wgi_cols if c in data.columns]
        if present:
            data["governance_composite"] = data[present].mean(axis=1)
            data["governance_x_complexity"] = (
                data["governance_composite"] * data["complexity"]
            )

    return data


def get_feature_cols(data: pd.DataFrame) -> list[str]:
    """Return feature columns matching C_run_GLM_nobayes.py."""
    base_cols = [
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
        "log_planned_expenditure",
        "expenditure_per_year_log",
        "governance_x_complexity",
        # "is_ngo_impl",
        *[f"rep_org_{i}" for i in range(NUM_ORGS_KEEP - 1)],
        "umap3_x",
        "umap3_y",
        "umap3_z",
        "sector_distance",
        "country_distance",
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
    if ADD_RATINGS and "rating" in data.columns:
        base_cols = ["rating"] + base_cols

    # Sector cluster columns (dynamic)
    sector_cluster_cols = [c for c in data.columns if c.startswith("sector_cluster_")]
    all_cols = base_cols + sector_cluster_cols

    # Only keep columns that exist
    available = [c for c in all_cols if c in data.columns]
    missing = [c for c in all_cols if c not in data.columns]
    if missing:
        print(f"Note: {len(missing)} feature cols not found in data: {missing[:10]}...")
    return available


def add_rf_llm_residual_corrector(
    data: pd.DataFrame,
    y: pd.Series,
    *,
    train_idx: pd.Index,
    pred_col: str,
    llm_col: str,
    out_col: str,
    alpha: float = 5.0,
    lam: float = 1.0,
    clip_lo: float = 0.0,
    clip_hi: float = 1.0,
) -> Ridge | None:
    """Fit a Ridge residual corrector using both RF prediction and LLM forecast prob.

    Fits:  r_i = beta_0 + beta_1 * rf_pred_i + beta_2 * llm_prob_i
    Applies to all activities where llm_prob is available; others fall back to rf_pred.

    Fitted on training activities where both pred_col and llm_col are non-null.
    This naturally handles recency since LLM forecasts tend to cover more recent activities.
    """
    feats = [pred_col, llm_col]
    valid_mask = data[feats].notna().all(axis=1)

    train_idx = pd.Index(train_idx)
    fit_mask = valid_mask.loc[train_idx] & y.loc[train_idx].notna()
    fit_idx = train_idx[fit_mask.to_numpy()]

    if len(fit_idx) == 0:
        print(
            f"    llm corrector: no rows to fit for {pred_col} + {llm_col}, skipping."
        )
        data[out_col] = data[pred_col].astype(float)
        return None

    X = data.loc[fit_idx, feats].astype(float).to_numpy()
    r = (
        y.loc[fit_idx].astype(float) - data.loc[fit_idx, pred_col].astype(float)
    ).to_numpy()

    model = Ridge(alpha=alpha, fit_intercept=True).fit(X, r)

    # initialise with RF pred, then overwrite where LLM coverage exists
    data[out_col] = data[pred_col].astype(float)
    apply_idx = data.index[valid_mask]
    X_apply = data.loc[apply_idx, feats].astype(float).to_numpy()
    resid_hat = model.predict(X_apply)
    data.loc[apply_idx, out_col] = (
        data.loc[apply_idx, pred_col].astype(float) + lam * resid_hat
    ).clip(clip_lo, clip_hi)

    coef = model.coef_.ravel()
    print(
        f"    llm corrector: n_fit={len(fit_idx)}  "
        f"intercept={float(model.intercept_):+.4f}  "
        f"coef[rf]={float(coef[0]):+.4f}  coef[llm]={float(coef[1]):+.4f}"
    )
    return model


def _eval_ensemble_probas(
    probas_val: np.ndarray,
    y_val: np.ndarray,
    tag: str,
    model_type: str,
    n_train_pos: int,
    n_tr: int,
) -> dict:
    """Evaluate val-set probabilities and return a metrics dict."""
    y = y_val.astype(float)
    p = probas_val.astype(float)
    # Baseline = Brier score of always predicting train_rate on the actual val labels.
    # mean((train_rate - y_i)^2), NOT train_rate*(1-train_rate) -- those only agree when
    # train and val positive rates match.
    train_rate = n_train_pos / n_tr if n_tr > 0 else float(y.mean())
    result = {
        "tag": tag,
        "model_type": model_type,
        "train_n": n_tr,
        "train_n_pos": n_train_pos,
        "train_base_rate": train_rate,
        "val_n": len(y),
        "val_n_pos": int(y.sum()),
    }
    try:
        result["val_auc"] = float(roc_auc_score(y, p))
    except Exception:
        pass
    try:
        result["val_ap"] = float(average_precision_score(y, p))
    except Exception:
        pass
    try:
        result["val_brier"] = float(np.mean((p - y) ** 2))
        result["val_brier_skill"] = brier_skill_score(y, p, train_base_rate=train_rate)
        result["val_brier_base"] = float(np.mean((train_rate - y) ** 2))
    except Exception:
        pass
    try:
        result["val_pairwise_ordering_prob"] = float(pairwise_ordering_prob(y, p))
    except Exception:
        pass
    try:
        result["val_acc"] = side_accuracy(y, p, 0.5)
    except Exception:
        pass
    result["val_y_true"] = y.tolist()
    result["val_y_pred"] = p.tolist()
    return result


def _pick_best_model(
    candidates: list[tuple[str, dict]],
    majority_acc: float,
) -> str:
    """
    Pick the best model from candidates using a balanced multi-metric criterion.

    Metrics compared: BrierSkill, POP (pairwise ordering prob), AccuracySkill.

    Hard criteria -- a candidate is ineligible to win if either:
      - val_brier_skill <= 0  (no calibration improvement over naive baseline)
      - val_acc - majority_acc <= 0  (no accuracy improvement over majority-class)
    If all candidates fail hard criteria, falls back to highest BrierSkill (safety net;
    _maybe_revert_const_base will handle the final quality gate downstream).

    Among eligible candidates, uses pairwise tournament comparison across 3 metrics.
    A small loss on one metric is forgiven if:
      1. The absolute loss < MAX_FORGIVE  (large losses are never forgiven)
      2. There is a compensating gain on another metric of > COMP_RATIO * loss
    A candidate beats another if it has more net wins (after forgiveness) than net losses.
    Tiebreak: highest BrierSkill.
    """
    METRICS = ["val_brier_skill", "val_pairwise_ordering_prob", "acc_skill"]
    MAX_FORGIVE = 0.015  # losses >= this are never forgiven (~1.5 percentage points)
    COMP_RATIO = 0.20  # compensating gain must be > 20% of the loss being forgiven

    def acc_skill(r: dict) -> float:
        acc = r.get("val_acc")
        return (acc - majority_acc) if acc is not None else float("-inf")

    def get_metrics(r: dict) -> dict:
        return {
            "val_brier_skill": r.get("val_brier_skill", float("-inf")),
            "val_pairwise_ordering_prob": r.get(
                "val_pairwise_ordering_prob", float("-inf")
            ),
            "acc_skill": acc_skill(r),
        }

    # Hard disqualification -- both must be positive to be eligible
    qualified = [
        (name, r)
        for name, r in candidates
        if r.get("val_brier_skill", float("-inf")) > 0.0
        and acc_skill(r) > ACC_SKILL_MIN_IMPROVEMENT
    ]
    pool = qualified if qualified else candidates  # safety fallback

    if len(pool) == 1:
        return pool[0][0]

    metrics_by_name = {name: get_metrics(r) for name, r in pool}
    win_counts: dict[str, int] = {name: 0 for name, _ in pool}

    for name_a, _ in pool:
        for name_b, _ in pool:
            if name_a == name_b:
                continue
            m_a, m_b = metrics_by_name[name_a], metrics_by_name[name_b]
            wins, losses = [], []
            for metric in METRICS:
                diff = m_a[metric] - m_b[metric]
                if diff > 0:
                    wins.append((metric, diff))
                elif diff < 0:
                    losses.append((metric, -diff))

            # Forgive small losses that have a sufficiently large compensating gain
            forgiven = sum(
                1
                for _, loss_delta in losses
                if loss_delta < MAX_FORGIVE
                and any(win_delta > COMP_RATIO * loss_delta for _, win_delta in wins)
            )
            if len(wins) > len(losses) - forgiven:
                win_counts[name_a] += 1

    best_wins = max(win_counts.values())
    tied = [name for name, w in win_counts.items() if w == best_wins]
    if len(tied) == 1:
        return tied[0]

    # Tiebreak: highest BrierSkill
    result_by_name = dict(pool)
    return max(
        tied, key=lambda n: result_by_name[n].get("val_brier_skill", float("-inf"))
    )


def _normalize_f_record(r: dict) -> dict:
    """Bridge F_ record fields to _pick_best_model's expected keys.

    F_ stores pairwise ordering prob as 'val_pop'; _pick_best_model reads
    'val_pairwise_ordering_prob'.  Copy the value so both keys are present.
    """
    r = dict(r)
    if "val_pop" in r and "val_pairwise_ordering_prob" not in r:
        r["val_pairwise_ordering_prob"] = r["val_pop"]
    return r


def load_per_tag_strategies(
    f_results_path: Path, feature_cols: list[str], _unused_models_path: Path
) -> dict[str, dict]:
    """
    Read F_regularization_strategies results and use _pick_best_model to select
    the best implementable RF feature-selection strategy per tag.

    Strategies considered (those D_ can implement): baseline, top5_feat,
    top10_feat, top30_feat.  heavy_reg / very_shallow are F_-only and skipped.

    majority_acc is derived from val_n / val_n_pos stored in F_ records.

    Returns dict: tag -> {"strategy": str, "feat_idx": list[int] | None}
    where feat_idx is None for "baseline" (use all features).
    """
    if not f_results_path.exists():
        print(
            f"  [per_tag_strategy] F results not found at {f_results_path} -- using baseline for all tags"
        )
        return {}

    with f_results_path.open() as fh:
        f_records = json.load(fh)

    # Index: tag -> strategy -> record
    f_idx: dict[str, dict[str, dict]] = {}
    for r in f_records:
        f_idx.setdefault(r["tag"], {})[r["strategy"]] = r

    feat_col_index = {name: i for i, name in enumerate(feature_cols)}

    # Only strategies that D_ can actually implement as RF feature subsets
    RF_STRATS = ["baseline", "top5_feat", "top10_feat", "top30_feat"]

    result: dict[str, dict] = {}
    for tag, strats in f_idx.items():
        baseline_rec = _normalize_f_record(strats.get("baseline", {}))
        if not baseline_rec.get("val_brier_skill"):
            result[tag] = {"strategy": "baseline", "feat_idx": None}
            continue

        # Compute majority_acc from stored val set composition
        val_n = baseline_rec.get("val_n", 0)
        val_n_pos = baseline_rec.get("val_n_pos", 0)
        majority_acc = max(val_n_pos, val_n - val_n_pos) / val_n if val_n > 0 else 0.5

        candidates = [
            (s, _normalize_f_record(strats[s])) for s in RF_STRATS if s in strats
        ]
        if not candidates:
            result[tag] = {"strategy": "baseline", "feat_idx": None}
            continue

        best_strat = _pick_best_model(candidates, majority_acc)

        if best_strat == "baseline":
            result[tag] = {"strategy": "baseline", "feat_idx": None}
        else:
            top_feat_names = strats[best_strat].get("top_features", [])
            feat_idx = [
                feat_col_index[n] for n in top_feat_names if n in feat_col_index
            ]
            if len(feat_idx) == 0:
                print(
                    f"  [per_tag_strategy] {tag}: top_features not found in current cols -- falling back to baseline"
                )
                result[tag] = {"strategy": "baseline", "feat_idx": None}
            else:
                result[tag] = {"strategy": best_strat, "feat_idx": feat_idx}

    return result


# ---------------------------------------------------------------------------
# Reusable helpers (importable by E_kfold_tag_assessment and other scripts)
# ---------------------------------------------------------------------------


def apply_start_year_correction(
    y_pred_all: np.ndarray,
    all_index: pd.Index,
    start_dates: pd.Series,
    tr_idx: pd.Index,
    y_tr: pd.Series,
) -> np.ndarray:
    """
    Fit Ridge(alpha=50) on residual ~ start_year using tr_idx rows, apply to all.
    Returns corrected probas (same length as all_index), clipped to [0, 1].

    Used by main() and importable by kfold scripts -- single source of truth for
    the start-year correction so any tuning (alpha, clipping) propagates everywhere.
    """
    pred_series = pd.Series(y_pred_all, index=all_index)
    start_years = start_dates.dt.year.astype(float)
    train_pred_s = pred_series.loc[y_tr.index]
    train_years_s = start_years.loc[y_tr.index]
    resid_s = y_tr.astype(float) - train_pred_s
    valid = train_years_s.notna() & resid_s.notna()
    zeros = np.zeros(len(all_index))
    if valid.sum() < 5:
        return y_pred_all, zeros
    yr_ridge = Ridge(alpha=START_YEAR_RIDGE_ALPHA, fit_intercept=True)
    yr_ridge.fit(
        train_years_s[valid].to_numpy().reshape(-1, 1), resid_s[valid].to_numpy()
    )
    all_years = start_years.reindex(all_index).fillna(start_years.median())
    correction = yr_ridge.intercept_ + yr_ridge.coef_[0] * all_years.to_numpy()
    return (pred_series.to_numpy() + correction).clip(0.0, 1.0), correction


def train_rf_et_ensemble(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_all: pd.DataFrame,
    n_train_pos: int | None = None,
    rf_params_override: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, object, object]:
    """
    Fit RF + ExtraTrees on X_tr/y_tr; return (rf_p, et_p, rf_clf, et_clf).

    Probas are for X_all.  The caller averages rf_p and et_p, optionally applying
    start-year correction to rf_p first (CORRECT_RF_BEFORE_ET).
    """
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

    meds = X_tr.median()
    X_tr_imp = X_tr.fillna(meds)
    X_all_imp = X_all.fillna(meds)

    pos_rate = float(y_tr.mean())
    cw = None if pos_rate > CLASS_WEIGHT_POS_RATE_THRESHOLD else "balanced"
    n_pos = n_train_pos if n_train_pos is not None else int(y_tr.sum())
    rf_params = (
        RF_PARAMS_RICH if n_pos >= ADAPTIVE_DEPTH_THRESHOLD else RF_PARAMS_SHALLOW
    )
    if rf_params_override:
        rf_params = {**rf_params, **rf_params_override}
    rf_params = {**rf_params, "class_weight": cw}

    rf_base = RandomForestClassifier(**rf_params)
    rf = rf_base
    rf.fit(X_tr_imp, y_tr)
    rf_p = rf.predict_proba(X_all_imp)[:, 1]

    if not hasattr(train_rf_et_ensemble, "_params_printed"):
        train_rf_et_ensemble._params_printed = True
        _keys = [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "max_samples",
            "ccp_alpha",
            "class_weight",
            "bootstrap",
        ]
        print(
            "  [RF effective params] "
            + "  ".join(f"{k}={rf_params.get(k)}" for k in _keys)
        )

    if ET_PARAMS_OVERRIDE is not None:
        et_kw = {
            k: v
            for k, v in ET_PARAMS_OVERRIDE.items()
            if k not in ("class_weight", "max_samples")
        }
        et_kw.update(
            {"class_weight": cw, "bootstrap": False, "random_state": 43, "n_jobs": -1}
        )
    else:
        et_kw = {
            "n_estimators": rf_params["n_estimators"],
            "max_depth": rf_params.get("max_depth", None),
            "min_samples_leaf": rf_params["min_samples_leaf"],
            "max_features": rf_params["max_features"],
            "class_weight": cw,
            "bootstrap": False,
            "random_state": 43,
            "n_jobs": -1,
        }
    if DISABLE_ET:
        return rf_p, rf_p, rf, rf

    et = ExtraTreesClassifier(**et_kw)
    et.fit(X_tr_imp, y_tr)
    et_p = et.predict_proba(X_all_imp)[:, 1]

    return rf_p, et_p, rf, et


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_n_estimators",
        type=int,
        default=None,
        help="If set, train RF/ET models with this many estimators instead of 500 (e.g. 100).",
    )
    parser.add_argument(
        "--add_ratings",
        type=lambda x: x.lower() != "false",
        default=None,
        help="Override ADD_RATINGS flag (true/false). Defaults to module-level ADD_RATINGS.",
    )
    parser.add_argument(
        "--param_suffix",
        type=str,
        default=None,
        help="If set, use the RF+ET param set from PARAM_SETS[suffix] and suffix all output paths.",
    )
    parser.add_argument(
        "--use_test",
        action="store_true",
        help="Train on train+val combined, evaluate on test set. Outputs suffixed with _test.",
    )
    parser.add_argument(
        "--nolimits",
        action="store_true",
        help=(
            "Remove all custom regularization: disable per-tag feature selection "
            "(USE_PER_TAG_STRATEGY=False), clear TAG_RF_PARAMS_OVERRIDES, and use "
            "the 45 non-noisy features for every tag with leaf=5, no depth limits. "
            "Outputs suffixed with _nolimits."
        ),
    )
    parser.add_argument(
        "--hardcode_tags",
        action="store_true",
        help=(
            "Use the hardcoded HARDCODED_14_TAGS list instead of loading val-run "
            "results to decide which tags to train. Implies --use_test behaviour "
            "for tag selection (no val-run JSON required)."
        ),
    )
    args = parser.parse_args()

    global RF_N_ESTIMATORS, ADD_RATINGS
    global RF_PARAMS_BASE, RF_PARAMS_RICH, RF_PARAMS_SHALLOW, ET_PARAMS_OVERRIDE
    global OUT_RESULTS, OUT_PREDICTIONS, OUT_MODELS, OUT_YEAR_CORR
    global USE_PER_TAG_STRATEGY, TAG_RF_PARAMS_OVERRIDES

    if args.add_ratings is not None:
        ADD_RATINGS = args.add_ratings
        print(f"[main] ADD_RATINGS overridden to {ADD_RATINGS}")

    RF_N_ESTIMATORS = (
        args.save_n_estimators if args.save_n_estimators is not None else 500
    )
    if args.save_n_estimators is not None:
        print(f"[main] RF/ET n_estimators overridden to {RF_N_ESTIMATORS}")
        RF_PARAMS_RICH["n_estimators"] = RF_N_ESTIMATORS
        RF_PARAMS_SHALLOW["n_estimators"] = RF_N_ESTIMATORS

    if args.param_suffix is not None:
        if args.param_suffix not in PARAM_SETS:
            raise ValueError(
                f"Unknown --param_suffix {args.param_suffix!r}. Valid: {sorted(PARAM_SETS)}"
            )
        rf_p, et_p = PARAM_SETS[args.param_suffix]
        RF_PARAMS_BASE = dict(rf_p)
        RF_PARAMS_RICH = RF_PARAMS_BASE
        RF_PARAMS_SHALLOW = RF_PARAMS_BASE
        ET_PARAMS_OVERRIDE = dict(et_p)
        s = f"_{args.param_suffix}"
        OUT_RESULTS = OUT_DIR / f"tag_model_results{s}.json"
        OUT_PREDICTIONS = OUT_DIR / f"tag_predictions{s}.csv"
        OUT_MODELS = OUT_DIR / f"tag_models{s}.pkl"
        OUT_YEAR_CORR = OUT_DIR / f"tag_year_correction_data{s}.pkl"
        print(
            f"[main] --param_suffix={args.param_suffix}: RF={RF_PARAMS_BASE}  ET={ET_PARAMS_OVERRIDE}"
        )
        print(f"[main] Output paths suffixed with '{s}'")

    if args.nolimits:
        USE_PER_TAG_STRATEGY = False
        TAG_RF_PARAMS_OVERRIDES = {}
        s_nl = "_nolimits"
        OUT_RESULTS = OUT_DIR / f"tag_model_results{s_nl}.json"
        OUT_PREDICTIONS = OUT_DIR / f"tag_predictions{s_nl}.csv"
        OUT_MODELS = OUT_DIR / f"tag_models{s_nl}.pkl"
        OUT_YEAR_CORR = OUT_DIR / f"tag_year_correction_data{s_nl}.pkl"
        print("[nolimits] USE_PER_TAG_STRATEGY=False, TAG_RF_PARAMS_OVERRIDES cleared.")
        print("[nolimits] All tags use 45 non-noisy features, leaf=5, no depth limits.")
        print(f"[nolimits] Output paths suffixed with '{s_nl}'")

    val_model_types: dict[str, str] = (
        {}
    )  # populated from val-run results when --use_test
    if args.use_test:
        if args.hardcode_tags:
            # Bypass val-run JSON: treat all 14 hardcoded tags as rf+ET
            val_model_types = {tag: "rf+ET" for tag in HARDCODED_14_TAGS}
            print(
                f"[hardcode_tags] Using {len(val_model_types)} hardcoded tags (no val-run JSON required)"
            )
        else:
            # Load val-run decisions BEFORE overriding output paths (OUT_RESULTS still points to val file)
            if not OUT_RESULTS.exists():
                print(
                    f"ERROR: val run results not found at {OUT_RESULTS}. Run D_train_staged.py (without --use_test) first."
                )
                return
            with OUT_RESULTS.open() as _f:
                _val_run = json.load(_f)
            val_model_types = {r["tag"]: r.get("model_type", "rf+ET") for r in _val_run}
            print(
                f"[use_test] Loaded {len(val_model_types)} val-run model type decisions (frozen for test run)"
            )

        s_test = "_test" if not args.nolimits else "_nolimits_test"
        OUT_RESULTS = OUT_DIR / f"tag_model_results{s_test}.json"
        OUT_PREDICTIONS = OUT_DIR / f"tag_predictions{s_test}.csv"
        OUT_MODELS = OUT_DIR / f"tag_models{s_test}.pkl"
        OUT_YEAR_CORR = OUT_DIR / f"tag_year_correction_data{s_test}.pkl"
        print("WARNING: TEST SET Run! Training on train+val, evaluating on test set.")
        print(f"[use_test] Output paths suffixed with '{s_test}'")

    if not APPLIED_TAGS.exists():
        print(f"ERROR: Applied tags not found at {APPLIED_TAGS}")
        print("Run C_apply_tags_at_scale.py first.")
        return

    print("=" * 60)
    print("TRACK 1: Tag prediction models")
    print("=" * 60)

    # Load tags
    print(f"\nLoading applied tags from {APPLIED_TAGS} ...")
    tags_df, model_cols = load_applied_tags(APPLIED_TAGS)

    # Build feature matrix
    print("\nComputing training activity IDs for leak-free feature fitting...")
    feat_cutoff = LATEST_VALIDATION_POINT if args.use_test else None
    train_activity_ids = get_train_activity_ids_from_dates(
        str(INFO_FOR_ACTIVITY_FORECASTING),
        cutoff_date=feat_cutoff,
        keep_orgs=KEEP_REPORTING_ORGS,
    )
    print(f"  {len(train_activity_ids)} training activities identified")

    print("\nBuilding feature matrix...")
    data = build_feature_matrix(train_activity_ids=train_activity_ids)

    # Load LLM forecast probabilities (produced by E_llm_tag_forecasts.py)
    if ADD_LLM_CORRECTION:
        if LLM_FORECAST_PROBS.exists():
            print(f"\nLoading LLM forecast probabilities from {LLM_FORECAST_PROBS} ...")
            llm_probs_df = pd.read_csv(
                str(LLM_FORECAST_PROBS),
                index_col="activity_id",
                dtype={"activity_id": str},
            )
            llm_probs_df = llm_probs_df.drop(columns=["split"], errors="ignore")
            llm_probs_df = llm_probs_df.add_prefix(
                "llm_"
            )  # tag_xxx -> llm_tag_xxx (avoids collision with tag label cols)
            data = data.join(llm_probs_df, how="left")
            print(
                f"  Loaded {len(llm_probs_df.columns)} LLM probability columns for {llm_probs_df.notna().any(axis=1).sum()} activities"
            )
        else:
            print(
                f"\nWARNING: ADD_LLM_CORRECTION=True but {LLM_FORECAST_PROBS} not found."
            )
            print(
                "  Run E_llm_tag_forecasts.py first. Falling back to RF-only predictions."
            )

    # Merge tags into data -- inner join to exclude activities that were never tagged.
    # Untagged activities have no ground truth; filling with 0 would be wrong.
    data = data.join(tags_df, how="inner")
    for col in model_cols:
        if col not in data.columns:
            data[col] = np.nan

    # Assert: unsigned tag columns must have no NaN after the inner join.
    # NaN would mean an activity was processed by the LLM but a tag was never
    # assessed -- it would get assumed 0 downstream, which is wrong.
    unsigned_tag_cols = [
        c
        for c in model_cols
        if not c.endswith("_attempted") and not c.endswith("_success")
    ]
    _unsigned_nulls = {
        c: int(data[c].isna().sum()) for c in unsigned_tag_cols if data[c].isna().any()
    }
    if _unsigned_nulls:
        raise AssertionError(
            f"Unsigned tag columns have NaN after inner join -- these activities were "
            f"never assessed for these tags and must not be assumed 0:\n{_unsigned_nulls}"
        )

    # Add start_date for splitting
    if "start_date" not in data.columns:
        data["start_date"] = data.apply(pick_start_date, axis=1)

    # Restrict to completed activities only (matching A_overall_rating_fit_and_evaluate.py)
    data = data[data["is_completed"].fillna(0).astype(int) == 1].copy()
    print(f"\nData shape after is_completed filter: {data.shape}")

    # Split -- outcome_tags has a smaller universe (tagged activities only) so
    # we do not assert against the overall_ratings canonical split.
    train_idx, val_idx, _test_idx = split_latest_by_date_with_cutoff(data, "start_date")

    # Save split IDs to the shared eval-set-sizes directory for cross-script comparison
    _eval_dir = DATA_DIR / "eval_set_sizes"
    _eval_dir.mkdir(parents=True, exist_ok=True)
    _tags_splits = pd.concat(
        [
            pd.DataFrame(
                {"activity_id": pd.Index(train_idx).astype(str), "split": "train"}
            ),
            pd.DataFrame(
                {"activity_id": pd.Index(val_idx).astype(str), "split": "val"}
            ),
            pd.DataFrame(
                {"activity_id": pd.Index(_test_idx).astype(str), "split": "test"}
            ),
        ],
        ignore_index=True,
    ).sort_values(["split", "activity_id"])
    _tags_splits.to_csv(_eval_dir / "outcome_tags_splits.csv", index=False)
    print(
        f"[eval_set_sizes] outcome_tags splits saved to {_eval_dir / 'outcome_tags_splits.csv'}"
    )

    if EXCLUDE_TEST_LEAKAGE_RISK:
        _test_idx = _test_idx.difference(pd.Index(list(TEST_LEAKAGE_RISK_IDS)))

    if args.use_test:
        train_idx = train_idx.append(val_idx)
        val_idx = _test_idx
        print(
            f"[use_test] Split: train+val={len(train_idx)}, eval=test ({len(val_idx)})"
        )
    else:
        print(
            f"Split sizes: train={len(train_idx)}, val={len(val_idx)} (test held out)"
        )

    # Feature columns
    feature_cols = get_feature_cols(data)
    print(f"Using {len(feature_cols)} base features")

    if DROP_NOISY_FEATURE_GROUPS:
        noisy_set = set(NOISY_FEATURE_GROUPS)
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if c not in noisy_set]
        dropped = before - len(feature_cols)
        print(
            f"[DROP_NOISY_FEATURE_GROUPS] Dropped {dropped} noisy features, {len(feature_cols)} remaining"
        )

    _feat_save_dir = DATA_DIR / "feature_lists"
    _feat_save_dir.mkdir(parents=True, exist_ok=True)
    _feat_label = "outcome_tag_nolimits" if args.nolimits else "outcome_tag"
    _feat_save_path = _feat_save_dir / f"{_feat_label}_features.json"
    with open(_feat_save_path, "w") as _f:
        json.dump(
            {
                "model": _feat_label,
                "n_features": len(feature_cols),
                "features": feature_cols,
            },
            _f,
            indent=2,
        )
    print(f"[feature_lists] Saved {len(feature_cols)} features to {_feat_save_path}")

    X_train = data.loc[train_idx, feature_cols]
    X_val = data.loc[val_idx, feature_cols]

    # ---- Verbose pre-training inspection ----
    if VERBOSE:
        W = 70
        print("\n" + "=" * W)
        print("VERBOSE: DATA STRUCTURE INSPECTION")
        print("=" * W)

        # Overall shape
        print("\n[Overall data shape]")
        print(f"  Total rows (after org restrict + merge): {len(data)}")
        print(f"  Total columns:                           {data.shape[1]}")
        print(f"  Index name:                              {data.index.name}")

        # Split summary with date ranges
        print("\n[Split summary]")
        for split_name, idx in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", _test_idx),
        ]:
            if len(idx) > 0:
                dates = data.loc[idx, "start_date"].dropna()
                date_range = (
                    f"{dates.min().date()} -> {dates.max().date()}"
                    if len(dates) > 0
                    else "N/A"
                )
            else:
                date_range = "N/A"
            print(
                f"  {split_name:6s}: {len(idx):5d} rows   start_date range: {date_range}"
            )
        print(
            f"  cutoff: train <= {LATEST_TRAIN_POINT}  |  val <= {LATEST_VALIDATION_POINT}  |  test < {TOO_LATE_CUTOFF}"
        )

        # Reporting org breakdown
        if "reporting_orgs" in data.columns:
            print("\n[Reporting org counts (train / val / test)]")
            for org in KEEP_REPORTING_ORGS:
                tr = (data.loc[train_idx, "reporting_orgs"] == org).sum()
                vl = (data.loc[val_idx, "reporting_orgs"] == org).sum()
                te = (data.loc[_test_idx, "reporting_orgs"] == org).sum()
                print(f"  {org[:55]:55s}  tr={tr:4d}  vl={vl:4d}  te={te:4d}")

        # Feature columns detail
        print(f"\n[Feature columns ({len(feature_cols)} total)]")
        print(
            f"  {'feature':45s}  {'dtype':8s}  {'null%':>6s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}"
        )
        print(f"  {'-'*45}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for fc in feature_cols:
            col_data = data[fc]
            null_pct = col_data.isna().mean() * 100
            dtype_str = str(col_data.dtype)[:8]
            if pd.api.types.is_numeric_dtype(col_data):
                mean_val = col_data.mean()
                std_val = col_data.std()
                min_val = col_data.min()
                max_val = col_data.max()
                print(
                    f"  {fc:45s}  {dtype_str:8s}  {null_pct:5.1f}%  {mean_val:8.3f}  {std_val:8.3f}  {min_val:8.3f}  {max_val:8.3f}"
                )
            else:
                n_unique = col_data.nunique()
                print(
                    f"  {fc:45s}  {dtype_str:8s}  {null_pct:5.1f}%  (non-numeric, {n_unique} unique values)"
                )

        # Tag columns: counts per split
        print(
            f"\n[Tag columns: positives per split ({len(model_cols)} total tag columns)]"
        )
        print(
            f"  {'tag_column':50s}  {'total':>6s}  {'tr_pos':>6s}  {'tr_neg':>6s}  {'val':>6s}  {'test':>6s}"
        )
        print(f"  {'-'*50}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        skipped = 0
        for mc in model_cols:
            if mc not in data.columns:
                print(f"  {mc:50s}  MISSING FROM DATA")
                skipped += 1
                continue
            total_pos = int(data[mc].sum())
            tr_pos = int(data.loc[train_idx, mc].sum())
            tr_neg = len(train_idx) - tr_pos
            vl_pos = int(data.loc[val_idx, mc].sum())
            te_pos = int(data.loc[_test_idx, mc].sum())
            if tr_pos < MIN_TAG_TRAIN_COUNT_LOW:
                skipped += 1
            print(
                f"  {mc:50s}  {total_pos:6d}  {tr_pos:6d}  {tr_neg:6d}  {vl_pos:6d}  {te_pos:6d}"
            )
        print(f"\n  Summary: Skipped(< {MIN_TAG_TRAIN_COUNT_LOW} train pos)={skipped}")

        # Columns in data not used as features or tags
        non_feature_non_tag = [
            c
            for c in data.columns
            if c not in feature_cols
            and c not in model_cols
            and c not in ("start_date", "rating", "reporting_orgs")
        ]
        print(
            f"\n[Other columns in data not used as features or tags ({len(non_feature_non_tag)})]"
        )
        print(f"  {non_feature_non_tag}")

        print("\n" + "=" * W)

    if EXCLUDE_ATTEMPTED_TAGS:
        before = len(model_cols)
        model_cols = [c for c in model_cols if "attempted" not in c]
        print(
            f"[EXCLUDE_ATTEMPTED_TAGS] Dropped {before - len(model_cols)} _attempted columns, {len(model_cols)} remaining"
        )

    # Save full list before balanced-only filter (needed for imbalanced loop below)
    full_model_cols = list(model_cols)

    if args.hardcode_tags and not args.use_test:
        # Standalone --hardcode_tags (no --use_test): restrict to the 14 hardcoded tags
        before = len(model_cols)
        model_cols = [c for c in model_cols if c in HARDCODED_14_TAGS]
        full_model_cols = [c for c in full_model_cols if c in HARDCODED_14_TAGS]
        print(
            f"[hardcode_tags] Restricted model_cols to 14 hardcoded tags: {before} -> {len(model_cols)}"
        )
    elif args.use_test and val_model_types:
        before = len(model_cols)
        model_cols = [c for c in model_cols if c in val_model_types]
        full_model_cols = [c for c in full_model_cols if c in val_model_types]
        label = "hardcoded" if args.hardcode_tags else "val-run"
        print(
            f"[use_test] Restricted model_cols to {label} tags: {before} -> {len(model_cols)}"
        )

    # Train one LightGBM + one RF per model column
    all_results = []
    trained_models = {}  # col -> {"lgbm": clf, "rf": clf}
    all_probas = {}  # f"{col}__lgbm" / f"{col}__rf" -> pd.Series of probas
    year_corr_data: dict[str, dict] = (
        {}
    )  # col -> {"uncorrected": Series, "correction": ndarray}

    train_medians = X_train.median()
    X_all = data[feature_cols].fillna(train_medians)

    # Per-tag strategy (feature selection based on F regularization results)
    per_tag_strategies: dict[str, dict] = {}
    if USE_PER_TAG_STRATEGY:
        print("\n[per_tag_strategy] Loading per-tag strategies from F results...")
        per_tag_strategies = load_per_tag_strategies(
            OUT_REGULARIZATION, feature_cols, OUT_MODELS
        )  # OUT_MODELS unused but kept for signature compat
        strategy_counts = {}
        for v in per_tag_strategies.values():
            strategy_counts[v["strategy"]] = strategy_counts.get(v["strategy"], 0) + 1
        print(f"  Strategies assigned: {strategy_counts}")

    for col in model_cols:
        if col not in data.columns:
            continue

        y_train_col = data.loc[train_idx, col].dropna()
        y_val_col = data.loc[val_idx, col].dropna()

        n_train_pos = int(y_train_col.sum())
        n_val_pos = int(y_val_col.sum())

        if n_train_pos < MIN_TAG_TRAIN_COUNT_LOW:
            print(
                f"  SKIP [{col}]: only {n_train_pos} positives in train (min={MIN_TAG_TRAIN_COUNT_LOW})"
            )
            continue

        is_summary = not (col.endswith("_attempted") or col.endswith("_success"))
        type_label = "summ" if is_summary else "outc"

        n_tr = len(y_train_col)  # applicable train rows only
        n_vl = len(y_val_col)  # applicable val rows only
        tag_strat_preview = per_tag_strategies.get(col, {}).get("strategy", "baseline")
        feat_k = len(per_tag_strategies.get(col, {}).get("feat_idx") or []) or "all"
        print(
            f"\n  [{type_label}] {col}  [strategy={tag_strat_preview}, features={feat_k}]\n"
            f"    train: N={n_tr}  pos={n_train_pos}  neg={n_tr - n_train_pos}\n"
            f"    val:   N={n_vl}  pos={n_val_pos}  neg={n_vl - n_val_pos}"
        )

        trained_models[col] = {}

        rf_params = {"n_estimators": RF_N_ESTIMATORS}
        depth_label = "default"

        # For heavily positive-skewed tags (>65% positive), class_weight="balanced" forces
        # the model to predict near 0.5.  Using class_weight=None lets the model predict
        # near the natural base rate, so calibration only needs to make small corrections.
        pos_rate_train = n_train_pos / n_tr if n_tr > 0 else 0.0
        if pos_rate_train > CLASS_WEIGHT_POS_RATE_THRESHOLD:
            rf_params = {**(rf_params or {}), "class_weight": None}
            depth_label += "+cw=none"

        # Per-tag feature selection (top-k features from F regularization results)
        # X is sliced to y's index so rows match (NaN targets were dropped above)
        tag_strat = per_tag_strategies.get(
            col, {"strategy": "baseline", "feat_idx": None}
        )
        feat_idx = tag_strat["feat_idx"]  # None = use all features
        tag_strategy_label = tag_strat["strategy"]
        if col in TAGS_SKIP_FEATURE_SELECTION:
            feat_idx = None  # force all features for this tag
        if feat_idx is not None:
            active_feat_cols = [feature_cols[i] for i in feat_idx]
            # Force rating into every per-tag subselected feature set when ADD_RATINGS is on
            if (
                ADD_RATINGS
                and "rating" in feature_cols
                and "rating" not in active_feat_cols
            ):
                active_feat_cols = ["rating"] + active_feat_cols
            X_train_tag = X_train.loc[y_train_col.index, active_feat_cols]
            X_val.loc[y_val_col.index, active_feat_cols]
            X_all_tag = X_all[active_feat_cols]
            train_medians[active_feat_cols]
        else:
            active_feat_cols = feature_cols
            X_train_tag = X_train.loc[y_train_col.index]
            X_val.loc[y_val_col.index]
            X_all_tag = X_all

        # ---- Train RF+ET ensemble (single source of truth for hyperparams) ----
        rf_all_p, et_all_p, rf_clf, et_clf = train_rf_et_ensemble(
            X_train_tag,
            y_train_col,
            X_all_tag,
            n_train_pos=n_train_pos,
            rf_params_override=TAG_RF_PARAMS_OVERRIDES.get(col),
        )
        year_corr_vec = np.zeros(len(data))  # unscaled correction; zero if not applied

        # ---- Start-year linear trend correction (optional) ----
        if ADD_START_YEAR_CORRECTION and col not in SKIP_START_YEAR_CORRECTION_TAGS:
            if CORRECT_RF_BEFORE_ET:
                # Original behaviour: correct RF probas, then average with ET.
                # Net effect on ensemble = correction / 2.
                corrected_rf, year_corr_vec = apply_start_year_correction(
                    rf_all_p,
                    data.index,
                    data["start_date"],
                    train_idx,
                    y_train_col,
                )
                ens_all_p = (corrected_rf + et_all_p) / 2.0
            else:
                # Correct the averaged ensemble directly (full correction magnitude).
                ens_all_p_raw = (rf_all_p + et_all_p) / 2.0
                ens_all_p, year_corr_vec = apply_start_year_correction(
                    ens_all_p_raw,
                    data.index,
                    data["start_date"],
                    train_idx,
                    y_train_col,
                )
        else:
            ens_all_p = (rf_all_p + et_all_p) / 2.0

        ens_probas_s = pd.Series(ens_all_p, index=data.index)
        ens_val_p = ens_probas_s.loc[y_val_col.index].to_numpy().astype(float)

        # Store uncorrected probas and unscaled correction for sweep script
        year_corr_data[col] = {
            "uncorrected": pd.Series((rf_all_p + et_all_p) / 2.0, index=data.index),
            "correction": year_corr_vec,  # unscaled; apply multiplier * this
            "val_idx": val_idx,
        }

        # ---- Build result dict ----
        result = _eval_ensemble_probas(
            ens_val_p,
            y_val_col.to_numpy(),
            col,
            "rf",
            n_train_pos,
            n_tr,
        )
        # Train metrics
        X_tr_imp_ens = X_train_tag.fillna(X_train_tag.median())
        y_tr_arr = y_train_col.to_numpy().astype(float)
        ens_tr_p = (
            rf_clf.predict_proba(X_tr_imp_ens)[:, 1]
            + et_clf.predict_proba(X_tr_imp_ens)[:, 1]
        ) / 2
        try:
            result["train_auc"] = float(roc_auc_score(y_tr_arr, ens_tr_p))
        except Exception:
            pass
        try:
            result["train_brier"] = float(np.mean((ens_tr_p - y_tr_arr) ** 2))
            result["train_brier_skill"] = brier_skill_score(y_tr_arr, ens_tr_p)
        except Exception:
            pass
        try:
            result["train_pairwise_ordering_prob"] = float(
                pairwise_ordering_prob(y_tr_arr, ens_tr_p)
            )
        except Exception:
            pass
        try:
            result["train_acc"] = side_accuracy(y_tr_arr, ens_tr_p, 0.5)
        except Exception:
            pass

        result["model_type"] = "rf+ET"
        result["tag_strategy"] = tag_strategy_label
        all_results.append(result)
        all_probas[f"{col}__rf"] = ens_probas_s
        if (
            hasattr(rf_clf, "oob_decision_function_")
            and rf_clf.oob_decision_function_ is not None
        ):
            oob_s = pd.Series(np.nan, index=data.index)
            oob_s.loc[y_train_col.index] = rf_clf.oob_decision_function_[:, 1]
            all_probas[f"{col}__rf_oob"] = oob_s
        trained_models[col]["rf"] = rf_clf
        trained_models[col]["extra"] = et_clf
        trained_models[col]["feature_cols"] = active_feat_cols

        def _fmt(v, fmt=".3f"):
            return f"{v:{fmt}}" if v is not None else "N/A"

        print(
            f"    [rf+ET/{depth_label}]  "
            f"train: acc={_fmt(result.get('train_acc'), '.1%')}  POP={_fmt(result.get('train_pairwise_ordering_prob'))}  |  "
            f"val: acc={_fmt(result.get('val_acc'), '.1%')}  POP={_fmt(result.get('val_pairwise_ordering_prob'))}  "
            f"AUC={_fmt(result.get('val_auc'))}  AP={_fmt(result.get('val_ap'))}  BrierSkill={_fmt(result.get('val_brier_skill'))}"
        )

        # ---- Report chosen RF+ET model ----
        _rf_res = (
            all_results[-1]
            if all_results and all_results[-1].get("tag") == col
            else None
        )
        if _rf_res is not None:
            mt = _rf_res.get("model_type", "rf")
            bs = _rf_res.get("val_brier_skill", float("-inf"))
            pop = _rf_res.get("val_pairwise_ordering_prob", float("-inf"))
            print(f"  +- CHOSEN: {mt}  (BrierSkl {bs:+.4f}, POP {pop:.3f})")

        # ---- Model selection: frozen from val run (--use_test) or apply gate (val run) ----
        _chosen = (
            all_results[-1]
            if all_results and all_results[-1].get("tag") == col
            else None
        )
        if _chosen is not None and _chosen.get("model_type") != "const_base":
            if args.use_test:
                # Freeze model type from val run -- never re-evaluate gate on test data
                frozen_mt = val_model_types.get(col, "rf+ET")
                if frozen_mt == "const_base" and col in TAGS_SKIP_CONST_BASE_GATE:
                    frozen_mt = "rf+ET"
                    print(
                        "  +- OVERRIDING frozen const_base -> rf+ET (TAGS_SKIP_CONST_BASE_GATE)"
                    )
                if frozen_mt == "const_base":
                    base_rate = float(y_train_col.mean())
                    y_vl_arr = y_val_col.to_numpy(dtype=float)
                    n_vl_pos = int(y_vl_arr.sum())
                    _cb_brier_base = float(np.mean((base_rate - y_vl_arr) ** 2))
                    cb_res = {
                        "tag": col,
                        "model_type": "const_base",
                        "train_n": _chosen.get("train_n"),
                        "train_n_pos": _chosen.get("train_n_pos"),
                        "val_n": len(y_vl_arr),
                        "val_n_pos": n_vl_pos,
                        "val_acc": float(max(base_rate, 1 - base_rate)),
                        "val_auc": 0.5,
                        "val_brier_skill": 0.0,
                        "val_pairwise_ordering_prob": 0.5,
                        "val_brier": _cb_brier_base,
                        "val_brier_base": _cb_brier_base,
                        "val_y_true": y_vl_arr.tolist(),
                        "val_y_pred": np.full(len(y_vl_arr), base_rate).tolist(),
                    }
                    if n_vl_pos > 0 and n_vl_pos < len(y_vl_arr):
                        try:
                            cb_res["val_ap"] = float(
                                average_precision_score(
                                    y_vl_arr, np.full(len(y_vl_arr), base_rate)
                                )
                            )
                        except ValueError:
                            cb_res["val_ap"] = float(
                                "nan"
                            )  # e.g. only one class present
                    if f"{col}__rf" in all_probas:
                        del all_probas[f"{col}__rf"]
                    all_probas[f"{col}__const_base"] = pd.Series(
                        np.full(len(data), base_rate), index=data.index
                    )
                    all_results[-1] = cb_res
                    trained_models[col] = {"base_rate": base_rate}
                    print("  +- FROZEN: const_base (val-run decision)")
                else:
                    print(f"  +- FROZEN: {frozen_mt} (val-run decision)")
            else:
                # Val run: apply gate -- revert to const_base if model is too weak
                # Revert unless BOTH: BrierSkl > 0 AND POP > 50%.
                # Tags in TAGS_SKIP_CONST_BASE_GATE always keep their RF+ET model.
                _bs = _chosen.get("val_brier_skill", -1.0)
                _revert_reason = None
                if col in TAGS_SKIP_CONST_BASE_GATE:
                    pass  # never revert
                elif _bs <= 0:
                    _revert_reason = f"BrierSkl={_bs:+.4f} <= 0"
                elif _chosen.get("val_pairwise_ordering_prob", 1.0) <= 0.5:
                    _revert_reason = (
                        f"POP={_chosen.get('val_pairwise_ordering_prob'):.4f} <= 0.50"
                    )
                if _revert_reason:
                    base_rate = float(y_train_col.mean())
                    y_vl_arr = y_val_col.to_numpy(dtype=float)
                    n_vl_pos = int(y_vl_arr.sum())
                    _cb_brier_base = float(np.mean((base_rate - y_vl_arr) ** 2))
                    cb_res = {
                        "tag": col,
                        "model_type": "const_base",
                        "train_n": _chosen.get("train_n"),
                        "train_n_pos": _chosen.get("train_n_pos"),
                        "val_n": len(y_vl_arr),
                        "val_n_pos": n_vl_pos,
                        "val_acc": float(max(base_rate, 1 - base_rate)),
                        "val_auc": 0.5,
                        "val_brier_skill": 0.0,
                        "val_pairwise_ordering_prob": 0.5,
                        "val_brier": _cb_brier_base,
                        "val_brier_base": _cb_brier_base,
                        "val_y_true": y_vl_arr.tolist(),
                        "val_y_pred": np.full(len(y_vl_arr), base_rate).tolist(),
                    }
                    if n_vl_pos > 0 and n_vl_pos < len(y_vl_arr):
                        try:
                            cb_res["val_ap"] = float(
                                average_precision_score(
                                    y_vl_arr, np.full(len(y_vl_arr), base_rate)
                                )
                            )
                        except ValueError:
                            cb_res["val_ap"] = float(
                                "nan"
                            )  # e.g. only one class present
                    if f"{col}__rf" in all_probas:
                        del all_probas[f"{col}__rf"]
                    all_probas[f"{col}__const_base"] = pd.Series(
                        np.full(len(data), base_rate), index=data.index
                    )
                    all_results[-1] = cb_res
                    trained_models[col] = {"base_rate": base_rate}
                    print(f"  +- CHOSEN: const_base  ({_revert_reason})")

        # ---- LLM mean-preserving correction (applied on top of final chosen model) ----
        if ADD_LLM_CORRECTION:
            llm_col_name = f"llm_{col}"
            if llm_col_name in data.columns and data[llm_col_name].notna().any():
                final_key = next(
                    (
                        k
                        for k in [f"{col}__rf", f"{col}__nw", f"{col}__const_base"]
                        if k in all_probas
                    ),
                    None,
                )
                _final_res = (
                    all_results[-1]
                    if all_results and all_results[-1].get("tag") == col
                    else None
                )
                if final_key is not None and _final_res is not None:
                    _pre_acc = _final_res.get("val_acc")
                    _pre_pop = _final_res.get("val_pairwise_ordering_prob")
                    _pre_bs = _final_res.get("val_brier_skill")
                    # 80/20 blend: keep model prediction where LLM is NaN, blend where available
                    base_probas = all_probas[final_key]
                    llm_corrected = base_probas.copy()
                    has_llm = data[llm_col_name].notna()
                    llm_corrected[has_llm] = (
                        0.8 * base_probas[has_llm]
                        + 0.2 * data.loc[has_llm, llm_col_name]
                    ).clip(0.0, 1.0)

                    y_val_col_llm = data.loc[val_idx, col].dropna()
                    lc_val = (
                        llm_corrected.loc[y_val_col_llm.index].to_numpy().astype(float)
                    )
                    y_val_arr_ = y_val_col_llm.to_numpy().astype(float)
                    # Baseline = Brier score of always predicting train_rate on actual val labels
                    _tr_n = _final_res.get("train_n", 1) or 1
                    _tr_n_pos = _final_res.get("train_n_pos", 0)
                    _tr_rate = _tr_n_pos / _tr_n
                    try:
                        _final_res["val_auc"] = float(roc_auc_score(y_val_arr_, lc_val))
                    except ValueError:
                        _final_res["val_auc"] = float("nan")  # only one class in y_true
                    try:
                        _final_res["val_ap"] = float(
                            average_precision_score(y_val_arr_, lc_val)
                        )
                    except ValueError:
                        _final_res["val_ap"] = float("nan")  # only one class in y_true
                    try:
                        _final_res["val_brier"] = float(
                            np.mean((lc_val - y_val_arr_) ** 2)
                        )
                        _final_res["val_brier_skill"] = brier_skill_score(
                            y_val_arr_, lc_val, train_base_rate=_tr_rate
                        )
                        _final_res["val_brier_base"] = float(
                            np.mean((_tr_rate - y_val_arr_) ** 2)
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Brier computation failed for {col}: {e}"
                        ) from e
                    try:
                        _final_res["val_pairwise_ordering_prob"] = float(
                            pairwise_ordering_prob(y_val_arr_, lc_val)
                        )
                    except ValueError:
                        _final_res["val_pairwise_ordering_prob"] = float("nan")
                    try:
                        _final_res["val_acc"] = side_accuracy(y_val_arr_, lc_val, 0.5)
                    except Exception as e:
                        raise RuntimeError(
                            f"side_accuracy failed for {col}: {e}"
                        ) from e
                    _final_res["val_y_true"] = y_val_arr_.tolist()
                    _final_res["val_y_pred"] = lc_val.tolist()
                    all_probas[final_key] = llm_corrected

                    def _fmt_llm(v, fmt=".3f"):
                        return f"{v:{fmt}}" if v is not None else "N/A"

                    print(
                        f"  +- [+llm_corr on {final_key.split('__')[1]}]\n"
                        f"       before:  acc={_fmt_llm(_pre_acc, '.1%')}  POP={_fmt_llm(_pre_pop)}  BrierSkill={_fmt_llm(_pre_bs)}\n"
                        f"       after:   acc={_fmt_llm(_final_res.get('val_acc'), '.1%')}  POP={_fmt_llm(_final_res.get('val_pairwise_ordering_prob'))}  BrierSkill={_fmt_llm(_final_res.get('val_brier_skill'))}"
                    )

    # ---- Manual factor blending ----
    apply_manual_factor_blend(
        data, train_idx, val_idx, feature_cols, train_medians, all_results, all_probas
    )

    # ---- Method comparison table ----
    all_methods = sorted(set(r.get("model_type", "?") for r in all_results))
    if len(all_methods) > 1:
        print(f"\n{'='*65}")
        print("METHOD COMPARISON  (mean val metrics, tags with >= 1 val positive)")
        print(
            f"{'method':22s}  {'n_tags':>7}  {'mean_AUC':>9}  {'mean_AP':>8}  {'mean_BrierSkl':>14}  {'mean_POP':>9}"
        )
        print("-" * 80)
        rf_mean_auc = None
        for method in all_methods:
            mrs = [
                r
                for r in all_results
                if r.get("model_type") == method and (r.get("val_n_pos") or 0) >= 1
            ]
            if not mrs:
                continue
            aucs = [r["val_auc"] for r in mrs if r.get("val_auc") is not None]
            aps = [r["val_ap"] for r in mrs if r.get("val_ap") is not None]
            bss = [
                r["val_brier_skill"]
                for r in mrs
                if r.get("val_brier_skill") is not None
            ]
            pops = [
                r["val_pairwise_ordering_prob"]
                for r in mrs
                if r.get("val_pairwise_ordering_prob") is not None
            ]
            m_auc = sum(aucs) / len(aucs) if aucs else float("nan")
            m_ap = sum(aps) / len(aps) if aps else float("nan")
            m_bs = sum(bss) / len(bss) if bss else float("nan")
            m_pop = sum(pops) / len(pops) if pops else float("nan")
            delta = (
                f"  Delta={m_auc - rf_mean_auc:+.4f}" if rf_mean_auc is not None else ""
            )
            if method == "rf":
                rf_mean_auc = m_auc
            print(
                f"  {method:20s}  {len(mrs):7d}  {m_auc:9.4f}  {m_ap:8.4f}  {m_bs:14.4f}  {m_pop:9.4f}{delta}"
            )

    # Save results
    n_trained = len([r for r in all_results if r])
    print(f"\n{'='*60}")
    print(f"Trained {n_trained} models")
    print("\nResults summary (sorted by val AUC):")
    results_sorted = sorted(
        all_results, key=lambda r: r.get("val_auc") or 0, reverse=True
    )
    header = f"{'T':>2s} {'mdl':>5s} {'kind':>4s} {'tag':42s} {'tr_pos':>6s} {'tr_neg':>6s} {'vl_pos':>6s} {'val_acc':>8s} {'val_AUC':>8s} {'val_AP':>7s} {'BrierSkl':>9s}"
    print(header)
    print("-" * len(header))
    for r in results_sorted:
        tag_name = r["tag"]
        kind = (
            "summ"
            if not (tag_name.endswith("_attempted") or tag_name.endswith("_success"))
            else "outc"
        )
        tr_neg_n = (r.get("train_n") or 0) - (r.get("train_n_pos") or 0)
        acc_str = (
            f"{r.get('val_acc'):.1%}" if r.get("val_acc") is not None else "     N/A"
        )
        auc_str = (
            f"{r.get('val_auc'):.3f}" if r.get("val_auc") is not None else "   N/A"
        )
        ap_str = f"{r.get('val_ap'):.3f}" if r.get("val_ap") is not None else "  N/A"
        bs_str = (
            f"{r.get('val_brier_skill'):.3f}"
            if r.get("val_brier_skill") is not None
            else "     N/A"
        )
        print(
            f"  {r.get('track','?'):>2} {r.get('model_type','?'):>5s} {kind:>4s} {tag_name:42s} "
            f"{r.get('train_n_pos', 0):>6d} "
            f"{tr_neg_n:>6d} "
            f"{r.get('val_n_pos', 0):>6d} "
            f"{acc_str:>8s} "
            f"{auc_str:>8s} {ap_str:>7s} {bs_str:>9s}"
        )

    # ---- Weighted accuracy summary: real models only (excl. const_base fallbacks) ----
    _wt_model = _wt_base = _total_w = 0.0
    for r in all_results:
        if r.get("model_type") == "const_base":
            continue
        _vn = r.get("val_n") or 0
        _vp = r.get("val_n_pos") or 0
        _acc = r.get("val_acc")
        if _vn > 0 and _acc is not None:
            _base_acc = max(_vp, _vn - _vp) / _vn
            _wt_model += _acc * _vn
            _wt_base += _base_acc * _vn
            _total_w += _vn
    if _total_w > 0:
        _mean_model = _wt_model / _total_w
        _mean_base = _wt_base / _total_w
        n_real = sum(
            1
            for r in all_results
            if r.get("model_type") != "const_base" and (r.get("val_n") or 0) > 0
        )
        eval_label = "test" if args.use_test else "val"
        print(f"\n{'='*60}")
        print(
            f"WEIGHTED ACCURACY SUMMARY  ({n_real} real-model tags, excl. const_base fallbacks)"
        )
        print(f"  Constant baseline (majority class): {_mean_base:.1%}")
        print(f"  Chosen model ({eval_label}):              {_mean_model:.1%}")
        print(f"  Improvement:                        {_mean_model - _mean_base:+.1%}")

    # -- WG-POP DEBUG BLOCK (D_train_staged) ----------------------------------
    _eval_label_d = "test" if args.use_test else "val"
    _eval_idx_d = (
        val_idx  # val_idx is already test_idx when --use_test (see split logic above)
    )
    _start_year_d = data.loc[_eval_idx_d, "start_date"].dt.year.fillna(-1).astype(int)
    _group_key_d = (
        data.loc[_eval_idx_d, "reporting_orgs"].fillna("unknown").astype(str)
        + "|||"
        + _start_year_d.astype(str)
    )
    _org_counts_d = data.loc[_eval_idx_d, "reporting_orgs"].value_counts().to_dict()
    _unique_grps_d = _group_key_d.nunique()
    print(f"\n{'='*70}")
    print("[WG-POP DEBUG] D_train_staged")
    print(f"  eval set          : {_eval_label_d}  n={len(_eval_idx_d)}")
    print(f"  train set size    : {len(train_idx)}")
    print(
        f"  train_activity_ids: {len(train_activity_ids)}  (feat-fitting cutoff={feat_cutoff})"
    )
    print(f"  n_estimators      : {RF_PARAMS_BASE['n_estimators']}")
    print(f"  features          : {len(feature_cols)}")
    print(
        f"  group key         : reporting_orgs + start_year  ({_unique_grps_d} unique groups)"
    )
    print(f"  org breakdown     : { {k[:30]: v for k, v in _org_counts_d.items()} }")
    print(f"  year range (eval) : {_start_year_d.min()}-{_start_year_d.max()}")
    print(f"  {'tag':<52s}  {'model':>10s}  {'wg_pop':>7s}")
    print(f"  {'-'*73}")
    _wg_vals_d = []
    for _r in sorted(all_results, key=lambda x: x.get("tag", "")):
        _tag = _r.get("tag", "")
        _mt = _r.get("model_type", "?")
        if _mt == "const_base" or _tag not in data.columns:
            continue
        _col = f"{_tag}__rf" if f"{_tag}__rf" in all_probas else None
        if _col is None:
            continue
        _yt = data.loc[_eval_idx_d, _tag].dropna()
        _yp = all_probas[_col].reindex(_yt.index)
        _grp = _group_key_d.reindex(_yt.index)
        _wg = _wg_pop_fn(_yt.to_numpy(float), _yp.to_numpy(float), _grp.to_numpy(str))[
            "prob"
        ]
        _wg_vals_d.append(_wg)
        print(f"  {_tag:<52s}  {_mt:>10s}  {_wg:7.3f}")
    import numpy as _np

    print(f"  {'-'*73}")
    print(f"  {'mean (non-const)':<52s}  {'':>10s}  {_np.nanmean(_wg_vals_d):7.3f}")
    print(f"{'='*70}")
    # -------------------------------------------------------------------------

    # Save JSON results
    with OUT_RESULTS.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {OUT_RESULTS}")

    # Save predictions CSV (probabilities per activity per tag)
    if all_probas:
        proba_df = pd.DataFrame(all_probas)
        proba_df.index.name = "activity_id"
        proba_df.to_csv(OUT_PREDICTIONS)
        print(f"Predictions saved to: {OUT_PREDICTIONS}")

    # Compute training base rate for every tag (used by webapp for reference lines)
    all_tag_cols = list(dict.fromkeys(model_cols + full_model_cols))
    tag_base_rates = {
        col: float(data.loc[train_idx, col].mean())
        for col in all_tag_cols
        if col in data.columns
    }

    # Save year-correction data for Z_sweep_year_correction.py
    import pickle

    with OUT_YEAR_CORR.open("wb") as f:
        pickle.dump(
            {
                "year_corr_data": year_corr_data,
                "current_scale": 1.0,  # multiplier currently baked into saved probas
                "activity_ids": list(data.index),
                "start_years": data["start_date"].dt.year.to_dict(),
                "val_idx": val_idx,
                "train_idx": train_idx,
            },
            f,
        )
    print(f"Year correction data saved to: {OUT_YEAR_CORR}")

    # Save models
    with OUT_MODELS.open("wb") as f:
        pickle.dump(
            {
                "models": trained_models,  # {col: {"lgbm": clf, "rf": clf}}
                "feature_cols": feature_cols,
                "model_cols": model_cols,
                "train_medians": train_medians.to_dict(),
                "tag_base_rates": tag_base_rates,
            },
            f,
        )
    print(f"Models saved to: {OUT_MODELS}")


if __name__ == "__main__":
    main()
