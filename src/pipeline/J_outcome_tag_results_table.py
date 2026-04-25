"""
Prints the LaTeX table of outcome tag validation results.

Feature importances are loaded from the 3-split stability data saved by
I_outcome_tag_shap_stability.py (shap_split_stability_data.pkl).
RF and ET SHAP values are averaged per split.

Only features that pass ALL of the following consistency criteria across
every split are shown in the table:
  - same sign in every split
  - ranked in the top 10 by |SHAP| in every split
  - at least 5% of total |SHAP| importance in every split
Features are ordered by mean |SHAP| across splits.

Feature abbreviations used in the Features column:
  $U_x$/$U_y$/$U_z$ -- UMAP embedding coordinates (semantic similarity space)
  dist      -- similarity distance group (sector_distance + country_distance combined)
  gdp       -- GDP per capita
  dur       -- planned duration
  exp       -- expenditure group (planned_expenditure, log_planned_expenditure, expenditure_per_year_log combined)
  scp       -- activity scope
  o1/o2     -- reporting org fixed effects
  tgt       -- LLM target rating
  ctx       -- LLM context rating
  rsk       -- LLM risk rating
  cplx      -- LLM complexity rating
  fin       -- LLM finance rating
  imp       -- implementer performance score
  loan      -- finance_is_loan indicator
  int       -- integratedness score
  ctry      -- country environment group (gdp_percap, cpia_score, all WGI indicators combined)
  EAP/AFE/LAC -- regional dummies
  sc:*      -- sector cluster membership
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
UTILS_DIR = SRC_DIR / "utils"
DATA_DIR = SRC_DIR.parent / "data"

for _p in [str(UTILS_DIR), str(SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from G_outcome_tag_train import (
    APPLIED_TAGS,
    CORRECT_RF_BEFORE_ET,
    INFO_FOR_ACTIVITY_FORECASTING,
    LATEST_TRAIN_POINT,
    LATEST_VALIDATION_POINT,
    MERGED_OVERALL_RATINGS,
    SKIP_START_YEAR_CORRECTION_TAGS,
    TOO_LATE_CUTOFF,
    load_applied_tags,
)
from H_outcome_tag_evaluate import TAG_GROUPS
from leakage_risk import EXCLUDE_TEST_LEAKAGE_RISK, TEST_LEAKAGE_RISK_IDS
from scoring_metrics import (
    brier_skill_score,
    within_group_pairwise_ordering_prob,
    within_group_spearman_correlation,
)

# -- Feature abbreviation map --------------------------------------------------
FEAT_ABBREV: dict[str, str] = {
    "umap3_x": r"$U_x$",
    "umap3_y": r"$U_y$",
    "umap3_z": r"$U_z$",
    "sector_distance": "sd",
    "country_distance": "cd",
    "gdp_percap": "gdp",
    "planned_duration": "dur",
    "planned_expenditure": "exp",
    "log_planned_expenditure": "lexp",
    "expenditure_per_year_log": "eyr",
    "activity_scope": "scp",
    "rep_org_0": "FCDO",
    "rep_org_1": "ADB",
    "rep_org_2": "WB",
    "targets": "tgt",
    "context": "ctx",
    "risks": "rsk",
    "complexity": "cplx",
    "finance": "fin",
    "implementer_performance": "imp",
    "finance_is_loan": "loan",
    "integratedness": "int",
    "region_EAP": "EAP",
    "region_AFE": "AFE",
    "region_LAC": "LAC",
    "region_SA": "SA",
    "region_MNA": "MNA",
    "region_ECA": "ECA",
    # Feature groups (collapsed in SHAP analysis)
    "expenditure_grp": "exp",
    "country_env": "ctry",
    "similarity_dist": "dist",
}

# Features that are functionally equivalent (different transforms / correlated proxies
# for the same underlying concept). In the SHAP consistency analysis, all members are
# collapsed into one virtual feature whose |SHAP| is the sum of its members'.
FEATURE_GROUPS: dict[str, list[str]] = {
    "expenditure_grp": [
        "planned_expenditure",
        "log_planned_expenditure",
        "expenditure_per_year_log",
    ],
    "country_env": [
        "gdp_percap",
        "cpia_score",
        "wgi_control_of_corruption_est",
        "wgi_government_effectiveness_est",
        "wgi_political_stability_est",
        "wgi_regulatory_quality_est",
        "wgi_rule_of_law_est",
    ],
    "similarity_dist": [
        "sector_distance",
        "country_distance",
    ],
}


def _abbrev(feat: str) -> str:
    if feat in FEAT_ABBREV:
        return FEAT_ABBREV[feat]
    if feat.startswith("sector_cluster_"):
        raw = feat[len("sector_cluster_") :]
        words = raw.replace("_", " ").split()
        abbr = "".join(w[0].upper() for w in words[:4])
        return f"sc:{abbr}"
    # Unknown feature: escape underscores so LaTeX doesn't error
    return feat.replace("_", r"\_")


# Ordered legend for the caption: (set of raw feature names, display abbrev, description).
# Only entries whose raw feature names appear in the table are included.
_CAPTION_LEGEND: list[tuple[set, str, str]] = [
    (
        {"umap3_x", "umap3_y", "umap3_z"},
        r"$U_x$/$U_y$/$U_z$",
        "UMAP semantic embedding coordinates",
    ),
    (
        {"similarity_dist", "sector_distance", "country_distance"},
        "dist",
        "similarity distance group (sector + country distance combined)",
    ),
    (
        {
            "country_env",
            "gdp_percap",
            "cpia_score",
            "wgi_control_of_corruption_est",
            "wgi_government_effectiveness_est",
            "wgi_political_stability_est",
            "wgi_regulatory_quality_est",
            "wgi_rule_of_law_est",
        },
        "ctry",
        "country environment group (GDP, CPIA, all WGI indicators combined)",
    ),
    ({"planned_duration"}, "dur", "planned duration"),
    (
        {
            "expenditure_grp",
            "planned_expenditure",
            "log_planned_expenditure",
            "expenditure_per_year_log",
        },
        "exp",
        "planned expenditure group (raw/log/log-per-year combined)",
    ),
    ({"activity_scope"}, "scp", "activity scope"),
    ({"rep_org_0"}, "FCDO", "FCDO reporting-org fixed effect"),
    ({"rep_org_1"}, "ADB", "ADB reporting-org fixed effect"),
    ({"rep_org_2"}, "WB", "World Bank reporting-org fixed effect"),
    ({"targets"}, "tgt", "LLM-graded target score"),
    ({"context"}, "ctx", "LLM-graded context score"),
    ({"risks"}, "rsk", "LLM-graded risk score"),
    ({"complexity"}, "cplx", "LLM-graded complexity score"),
    ({"finance"}, "fin", "LLM-graded finance score"),
    ({"implementer_performance"}, "imp", "implementer performance score"),
    ({"finance_is_loan"}, "loan", "finance-is-loan indicator"),
    ({"integratedness"}, "int", "integratedness score"),
    ({"region_EAP"}, "EAP", r"East Asia \& Pacific regional dummy"),
    ({"region_AFE"}, "AFE", r"Africa (Eastern \& Southern) regional dummy"),
    ({"region_LAC"}, "LAC", r"Latin America \& Caribbean regional dummy"),
    ({"region_SA"}, "SA", "South Asia regional dummy"),
    ({"region_MNA"}, "MNA", r"Middle East \& North Africa regional dummy"),
    ({"region_ECA"}, "ECA", r"Europe \& Central Asia regional dummy"),
]

_GREEN_HEX = "006400"
_RED_HEX = "8B0000"


def _sign_fmt(feat: str, sign_val: float) -> str:
    """Feature abbreviation with arrow decoration coloured by sign.
    Positive -> green overrightarrow; negative -> red overleftarrow.
    """
    arrow = r"\overrightarrow" if sign_val >= 0 else r"\overleftarrow"
    color = _GREEN_HEX if sign_val >= 0 else _RED_HEX
    abbr = _abbrev(feat)
    if abbr.startswith("$") and abbr.endswith("$"):
        inner = abbr[1:-1]
    else:
        inner = r"\text{" + abbr + "}"
    return "$" + arrow + r"{{\color[HTML]{" + color + "}" + inner + "}}$"


def get_consistent_features(
    feature_cols: list[str],
    split_abs_means: list,
    split_signed_means: list,
    full_abs_mean: list | None = None,
    full_signed_mean: list | None = None,
    top_k: int = 10,
    min_importance_frac: float = 0.05,
) -> list[tuple[str, float, float]]:
    """Return (feat_or_group, mean_abs_importance, sign) for features consistent across all splits.

    Members of FEATURE_GROUPS are collapsed into a single virtual feature before the
    consistency check: their |SHAP| values are summed and the sign is taken from the
    dominant (highest |SHAP|) member in each split.

    A feature/group is included only if in every split it:
      - has the same sign
      - ranks in the top top_k by |SHAP|
      - accounts for >= min_importance_frac of total |SHAP|

    If full_abs_mean / full_signed_mean are provided (SHAP from the model trained on the
    full training set), the feature must additionally pass the same importance fraction and
    sign criteria in that model.

    Result is sorted by mean abs importance descending.
    """
    import numpy as np

    # Map each index in feature_cols to a group name (if it belongs to one)
    feat_to_group: dict[int, str] = {}
    group_indices: dict[str, list[int]] = {}
    for gname, members in FEATURE_GROUPS.items():
        idxs = [i for i, f in enumerate(feature_cols) if f in members]
        if idxs:
            group_indices[gname] = idxs
            for idx in idxs:
                feat_to_group[idx] = gname

    # Build synthetic feature list: one entry per ungrouped feature, one per group
    synth_feats: list[str] = []
    synth_index_lists: list[list[int]] = []
    added_groups: set[str] = set()
    for i, feat in enumerate(feature_cols):
        if i in feat_to_group:
            gname = feat_to_group[i]
            if gname not in added_groups:
                synth_feats.append(gname)
                synth_index_lists.append(group_indices[gname])
                added_groups.add(gname)
        else:
            synth_feats.append(feat)
            synth_index_lists.append([i])

    n_synth = len(synth_feats)

    def _collapse(abs_raw, signed_raw) -> tuple:
        abs_arr = np.asarray(abs_raw)
        signed_arr = np.asarray(signed_raw)
        s_abs = np.zeros(n_synth)
        s_signed = np.zeros(n_synth)
        for j, idxs in enumerate(synth_index_lists):
            group_abs = abs_arr[idxs]
            s_abs[j] = group_abs.sum()
            dominant = idxs[int(np.argmax(group_abs))]
            s_signed[j] = float(np.sign(signed_arr[dominant]))
        return s_abs, s_signed

    # Compute synthetic abs/signed means per split
    synth_abs_list: list[np.ndarray] = []
    synth_signed_list: list[np.ndarray] = []
    for abs_raw, signed_raw in zip(split_abs_means, split_signed_means, strict=False):
        s_abs, s_signed = _collapse(abs_raw, signed_raw)
        synth_abs_list.append(s_abs)
        synth_signed_list.append(s_signed)

    # Full-model collapsed vectors (optional)
    full_s_abs: np.ndarray | None = None
    full_s_signed: np.ndarray | None = None
    if full_abs_mean is not None and full_signed_mean is not None:
        full_s_abs, full_s_signed = _collapse(full_abs_mean, full_signed_mean)

    # Consistency check on synthetic features
    split_topk: list[set[int]] = []
    split_fracs: list[np.ndarray] = []
    for s_abs in synth_abs_list:
        total = s_abs.sum()
        fracs = s_abs / total if total > 0 else np.zeros(n_synth)
        topk = set(int(i) for i in np.argsort(s_abs)[::-1][:top_k])
        split_topk.append(topk)
        split_fracs.append(fracs)

    # Full-model fraction vector
    full_fracs: np.ndarray | None = None
    if full_s_abs is not None:
        total = full_s_abs.sum()
        full_fracs = full_s_abs / total if total > 0 else np.zeros(n_synth)

    results = []
    for i, feat in enumerate(synth_feats):
        # Split criteria
        if not all(i in tk for tk in split_topk):
            continue
        if not all(float(fr[i]) >= min_importance_frac for fr in split_fracs):
            continue
        signs = [float(synth_signed_list[s][i]) for s in range(len(synth_abs_list))]
        if len(set(signs)) > 1:
            continue
        # Full-model criteria (additional gate)
        if full_fracs is not None:
            if float(full_fracs[i]) < min_importance_frac:
                continue
            if float(np.sign(full_s_signed[i])) != signs[0]:
                continue
        mean_abs = float(np.mean([am[i] for am in synth_abs_list]))
        results.append((feat, mean_abs, signs[0]))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _consistent_feat_cell(consistent_feats: list[tuple[str, float, float]]) -> str:
    """LaTeX string of consistent features with sign arrows, or em-dash if none."""
    if not consistent_feats:
        return r"\textemdash"
    return ", ".join(_sign_fmt(feat, sign) for feat, _, sign in consistent_feats)


# -- Table rows: (display label, tag key, group) -------------------------------
TABLE_ROWS: list[tuple[str, str, str]] = [
    # Finance & budget
    (
        "Funds Cancelled or Unutilized",
        "tag_funds_cancelled_or_unutilized",
        "Finance \\& Budget",
    ),
    ("Funds Reallocated", "tag_funds_reallocated", ""),
    ("High Disbursement", "tag_high_disbursement", ""),
    ("Improved Financial Performance", "tag_improved_financial_performance", ""),
    ("Over Budget", "tag_over_budget_success", ""),
    # Activity Rescoping
    ("Closing Date Extended", "tag_closing_date_extended", "Activity Rescoping"),
    ("Project Restructured", "tag_project_restructured", ""),
    ("Targets Revised", "tag_targets_revised", ""),
    # Process and Implementation Challenges
    (
        "Activities Not Completed",
        "tag_activities_not_completed",
        "Process \\& Implementation",
    ),
    ("Design or Appraisal Shortcomings", "tag_design_or_appraisal_shortcomings", ""),
    (
        "External Factors Affected Outcomes",
        "tag_external_factors_affected_outcomes",
        "",
    ),
    ("Implementation Delays", "tag_implementation_delays", ""),
    (
        "Monitoring \\& Evaluation Challenges",
        "tag_monitoring_and_evaluation_challenges",
        "",
    ),
    # Target Achievement
    (
        "Capacity Building Delivered",
        "tag_capacity_building_delivered_success",
        "Target Achievement",
    ),
    ("Energy Sector Improvements", "tag_energy_sector_improvements_success", ""),
    ("Gender-Equitable Outcomes", "tag_gender_equitable_outcomes_success", ""),
    (
        "High Beneficiary Satisfaction or Reach",
        "tag_high_beneficiary_satisfaction_or_reach_success",
        "",
    ),
    ("Improved Livelihoods", "tag_improved_livelihoods_success", ""),
    ("Improved Service Delivery", "tag_improved_service_delivery_success", ""),
    ("Infrastructure Completed", "tag_infrastructure_completed_success", ""),
    (
        "Policy and Regulatory Reforms",
        "tag_policy_regulatory_reforms_success_success",
        "",
    ),
    ("Private Sector Engagement", "tag_private_sector_engagement_success", ""),
    ("Targets Met or Exceeded", "tag_targets_met_or_exceeded_success", ""),
]


def _year_correction_slope(
    tag: str,
    year_corr_data: dict,
    activity_ids: list,
    start_years: dict,
) -> float | None:
    """Return effective slope (change in ensemble prob/year) of the start-year correction."""
    import numpy as np

    if tag in SKIP_START_YEAR_CORRECTION_TAGS:
        return None
    if tag not in year_corr_data:
        return None
    corr = year_corr_data[tag].get("correction")
    if corr is None:
        return None
    years = np.array([start_years.get(aid, np.nan) for aid in activity_ids])
    valid = np.isfinite(corr) & np.isfinite(years) & (corr != 0)
    if valid.sum() < 2:
        return None
    b, _ = np.polyfit(years[valid], corr[valid], 1)
    # correction was applied to RF before averaging with ET -> net effect on ensemble = b/2
    return float(b / 2.0) if CORRECT_RF_BEFORE_ET else float(b)


def _acc_ratio(r: dict) -> float:
    """Accuracy relative to the majority-class baseline, evaluated on val.

    The majority class is determined from train class distribution (base_rate).
    The baseline accuracy is how often the majority-class predictor is correct
    on the val set -- matching E_kfold_tag_assessment.py's _fold_metrics logic.
    """
    train_n = r.get("train_n") or 1
    train_pos = r.get("train_n_pos") or 0
    base_rate = train_pos / train_n  # majority class from train

    val_n = r.get("val_n") or 1
    val_pos = r.get("val_n_pos") or 0
    val_pos_rate = val_pos / val_n

    # If majority class is positive: always predict positive -> correct on val_pos_rate of val
    # If majority class is negative: always predict negative -> correct on 1-val_pos_rate of val
    majority_acc = val_pos_rate if base_rate >= 0.5 else (1.0 - val_pos_rate)

    acc = r.get("val_acc") or 0.0
    return acc / majority_acc if majority_acc > 0 else 1.0


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Print test-set results (uses _test suffixed files from G_outcome_tag_train --use_test).",
    )
    parser.add_argument(
        "--nolimits",
        action="store_true",
        help="Print nolimits run results (uses _nolimits suffixed files from G_outcome_tag_train --nolimits).",
    )
    args = parser.parse_args()

    OUT_TAGS_DIR = DATA_DIR / "outcome_tags"

    # Build suffix matching G_outcome_tag_train convention: _nolimits then _test
    suffix_parts = []
    if args.nolimits:
        suffix_parts.append("nolimits")
        print("[nolimits] Using nolimits run outputs.", flush=True, file=sys.stderr)
    if args.test:
        suffix_parts.append("test")
        print("WARNING: TEST SET Run!", flush=True, file=sys.stderr)
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    eval_set_label = "test" if args.test else "validation"
    results_path = OUT_TAGS_DIR / f"tag_model_results{suffix}.json"
    year_corr_path = OUT_TAGS_DIR / f"tag_year_correction_data{suffix}.pkl"
    predictions_path = OUT_TAGS_DIR / f"tag_predictions{suffix}.csv"

    _required = [
        (results_path, results_path.name),
        (year_corr_path, year_corr_path.name),
        (predictions_path, predictions_path.name),
    ]
    _missing = [(p, n) for p, n in _required if not p.exists()]
    if _missing:
        print("\n" + "=" * 70, file=sys.stderr)
        cmd_flags = " ".join(["--" + p for p in suffix_parts]).replace("_", "-")
        print(
            f"ERROR: Missing output files. Run G_outcome_tag_train.py {cmd_flags} first.",
            file=sys.stderr,
        )
        for p, _n in _missing:
            print(f"  MISSING: {p}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    # -- Load year-correction data ---------------------------------------------
    import pickle as _pickle

    with year_corr_path.open("rb") as _f:
        _yr_pkl = _pickle.load(_f)
    _year_corr_data = _yr_pkl.get("year_corr_data", {})
    _yr_activity_ids = _yr_pkl.get("activity_ids", [])
    _yr_start_years = _yr_pkl.get("start_years", {})

    # -- Load saved model results ----------------------------------------------
    with results_path.open() as f:
        raw_results = json.load(f)
    results: dict[str, dict] = {r["tag"]: r for r in raw_results}

    # -- Load split stability SHAP data (from I_outcome_tag_shap_stability) --
    SPLIT_DATA_PATH = OUT_TAGS_DIR / f"shap_split_stability_data{suffix}.pkl"
    split_data: dict = {}
    if SPLIT_DATA_PATH.exists():
        with SPLIT_DATA_PATH.open("rb") as _f2:
            split_data = _pickle.load(_f2)
    else:
        print(
            f"WARNING: {SPLIT_DATA_PATH} not found. "
            "Run I_outcome_tag_shap_stability.py first to generate split SHAP data.",
            file=sys.stderr,
        )

    # -- Diagnostic: top-5 importance % per tag (averaged across splits) ------
    if split_data:
        import numpy as _np2

        print(
            "\n-- Top-5 feature importance % per tag (mean across splits, after grouping) --",
            file=sys.stderr,
        )
        for _tag, _td in sorted(split_data.items()):
            _feat_cols = _td["feature_cols"]
            _abs_means = _td["split_abs_means"]
            # Collapse groups the same way get_consistent_features does
            _feat_to_group: dict[int, str] = {}
            _group_indices: dict[str, list[int]] = {}
            for _gname, _members in FEATURE_GROUPS.items():
                _idxs = [_i for _i, _f in enumerate(_feat_cols) if _f in _members]
                if _idxs:
                    _group_indices[_gname] = _idxs
                    for _idx in _idxs:
                        _feat_to_group[_idx] = _gname
            _synth_feats: list[str] = []
            _synth_idx_lists: list[list[int]] = []
            _added: set[str] = set()
            for _i, _f in enumerate(_feat_cols):
                if _i in _feat_to_group:
                    _gn = _feat_to_group[_i]
                    if _gn not in _added:
                        _synth_feats.append(_gn)
                        _synth_idx_lists.append(_group_indices[_gn])
                        _added.add(_gn)
                else:
                    _synth_feats.append(_f)
                    _synth_idx_lists.append([_i])
            _n_synth = len(_synth_feats)
            # Average importance % across splits
            _avg_pct = _np2.zeros(_n_synth)
            for _abs_raw in _abs_means:
                _abs_arr = _np2.asarray(_abs_raw)
                _s_abs = _np2.array(
                    [_abs_arr[_idxs].sum() for _idxs in _synth_idx_lists]
                )
                _total = _s_abs.sum()
                if _total > 0:
                    _avg_pct += _s_abs / _total * 100
            _avg_pct /= max(len(_abs_means), 1)
            _top5 = _np2.argsort(_avg_pct)[::-1][:5]
            _tag_short = _tag.replace("tag_", "")[:45]
            print(f"  {_tag_short:<45s}", file=sys.stderr, end="")
            for _j in _top5:
                print(
                    f"  {_synth_feats[_j][:18]}:{_avg_pct[_j]:5.1f}%",
                    file=sys.stderr,
                    end="",
                )
            print(file=sys.stderr)
        print(file=sys.stderr)

    # -- Build groups and sort by accuracy ratio -------------------------------
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    for display, tag, group in TABLE_ROWS:
        if group:
            groups.append((group, []))
        groups[-1][1].append((display, tag))

    def tag_ratio(tag: str) -> float:
        r = results.get(tag, {})
        if r.get("model_type", "const_base") == "const_base":
            return 0.0
        return _acc_ratio(r)

    for i, (gname, rows) in enumerate(groups):
        groups[i] = (gname, sorted(rows, key=lambda dt: tag_ratio(dt[1]), reverse=True))

    def group_mean_ratio(rows: list[tuple[str, str]]) -> float:
        ratios = [
            tag_ratio(tag)
            for _, tag in rows
            if results.get(tag, {}).get("model_type", "const_base") != "const_base"
        ]
        return sum(ratios) / len(ratios) if ratios else 0.0

    groups.sort(key=lambda g: group_mean_ratio(g[1]), reverse=True)

    # -- Load WG-POP and WG-Spearman scores via scoring_metrics -----------------
    wg_pop_by_tag: dict[str, float | None] = {}
    wg_spearman_by_tag: dict[str, float | None] = {}
    perm_p_by_tag: dict[str, float] = {}
    bss_perm_p_by_tag: dict[str, float] = {}

    _preds_df = pd.read_csv(
        str(predictions_path), index_col="activity_id", dtype={"activity_id": str}
    )
    _info = pd.read_csv(
        str(INFO_FOR_ACTIVITY_FORECASTING),
        usecols=[
            "activity_id",
            "txn_first_date",
            "actual_start_date",
            "original_planned_start_date",
            "reporting_orgs",
        ],
        dtype={"activity_id": str},
    )
    for _col in ["txn_first_date", "actual_start_date", "original_planned_start_date"]:
        _info[_col] = pd.to_datetime(_info[_col], errors="coerce")
    _info["start_date"] = (
        _info["actual_start_date"]
        .combine_first(_info["original_planned_start_date"])
        .combine_first(_info["txn_first_date"])
    )
    _info["start_year"] = _info["start_date"].dt.year
    _info = _info.set_index("activity_id").reindex(_preds_df.index)

    _too_late = _info[_info["start_date"] >= pd.to_datetime(TOO_LATE_CUTOFF)].index
    _remaining = _info.drop(index=_too_late, errors="ignore")
    _test_idx = _remaining[
        _remaining["start_date"] > pd.to_datetime(LATEST_VALIDATION_POINT)
    ].index
    if EXCLUDE_TEST_LEAKAGE_RISK:
        _test_idx = _test_idx.difference(pd.Index(list(TEST_LEAKAGE_RISK_IDS)))
    _remaining2 = _remaining.drop(index=_test_idx, errors="ignore")
    _val_idx = _remaining2[
        _remaining2["start_date"] > pd.to_datetime(LATEST_TRAIN_POINT)
    ].index
    _eval_idx = _test_idx if args.test else _val_idx

    _label_rows: dict = {}
    with APPLIED_TAGS.open() as _f:
        for _line in _f:
            _line = _line.strip()
            if _line:
                _obj = json.loads(_line)
                _aid = str(_obj.get("activity_id", ""))
                _row: dict = {}
                for _tag, _present in _obj.get("unsigned_tags", {}).items():
                    _row[f"tag_{_tag}"] = int(bool(_present))
                for _tag, _success in _obj.get("signed_tags", {}).items():
                    _row[f"tag_{_tag}_success"] = (
                        np.nan if _success is None else int(bool(_success))
                    )
                _label_rows[_aid] = _row
    _labels_df = pd.DataFrame.from_dict(_label_rows, orient="index")
    _unsigned_cols = [c for c in _labels_df.columns if not c.endswith("_success")]
    _labels_df[_unsigned_cols] = _labels_df[_unsigned_cols].fillna(0)
    _labels_eval = _labels_df.reindex(_eval_idx)

    _group_key = (
        _info.reindex(_eval_idx)["reporting_orgs"].fillna("unknown").astype(str)
        + "|||"
        + _info.reindex(_eval_idx)["start_year"].fillna(-1).astype(int).astype(str)
    )
    _grp_arr = _group_key.to_numpy(dtype=str)

    for _pred_col in _preds_df.columns:
        for _sfx in ("__rf", "__ridge", "__const_base"):
            if _pred_col.endswith(_sfx):
                _tag_key = _pred_col[: -len(_sfx)]
                break
        else:
            continue
        if "attempted" in _tag_key:
            continue
        if _tag_key not in _labels_eval.columns:
            continue
        _yt = _labels_eval[_tag_key].reindex(_group_key.index).to_numpy(dtype=float)
        _yp = (
            _preds_df.loc[_eval_idx, _pred_col]
            .reindex(_group_key.index)
            .to_numpy(dtype=float)
        )

        _pop_result = within_group_pairwise_ordering_prob(_yt, _yp, _grp_arr)
        _pop_val = _pop_result["prob"]
        wg_pop_by_tag[_tag_key] = float(_pop_val) if np.isfinite(_pop_val) else None

        # Permutation test: shuffle predictions 1000x to build null WG-POP distribution
        if np.isfinite(_pop_val):
            _rng_perm = np.random.default_rng(42)
            _null = np.array(
                [
                    within_group_pairwise_ordering_prob(
                        _yt, _rng_perm.permutation(_yp), _grp_arr
                    )["prob"]
                    for _ in range(1000)
                ]
            )
            _null = _null[np.isfinite(_null)]
            perm_p_by_tag[_tag_key] = (
                float(np.mean(_null >= _pop_val)) if len(_null) > 0 else 1.0
            )

        # BSS permutation test
        _base_rate = float(np.nanmean(_yt))
        _obs_bss = float(brier_skill_score(_yt, _yp, _base_rate))
        if np.isfinite(_obs_bss):
            _rng_bss = np.random.default_rng(42)
            _bss_null = np.array(
                [
                    brier_skill_score(_yt, _rng_bss.permutation(_yp), _base_rate)
                    for _ in range(1000)
                ]
            )
            _bss_null = _bss_null[np.isfinite(_bss_null)]
            bss_perm_p_by_tag[_tag_key] = (
                float(np.mean(_bss_null >= _obs_bss)) if len(_bss_null) > 0 else 1.0
            )

        _spr_result = within_group_spearman_correlation(_yt, _yp, _grp_arr)
        _rho = _spr_result["correlation"]
        wg_spearman_by_tag[_tag_key] = float(_rho) if np.isfinite(_rho) else None

    # -- Pre-pass: compute consistent features once, collect used abbreviations -
    consistent_feats_cache: dict[str, list] = {}
    used_raw_feats: set[str] = set()
    has_sector_cluster = False
    for _gname, _rows in groups:
        for _display, _tag in _rows:
            if results.get(_tag, {}).get("model_type", "const_base") == "const_base":
                continue
            _tag_sd = split_data.get(_tag)
            if _tag_sd is None:
                consistent_feats_cache[_tag] = []
                continue
            _cf = get_consistent_features(
                _tag_sd["feature_cols"],
                _tag_sd["split_abs_means"],
                _tag_sd["split_signed_means"],
                full_abs_mean=_tag_sd.get("full_abs_mean"),
                full_signed_mean=_tag_sd.get("full_signed_mean"),
            )
            consistent_feats_cache[_tag] = _cf
            for _feat, _, _ in _cf:
                if _feat.startswith("sector_cluster_"):
                    has_sector_cluster = True
                else:
                    used_raw_feats.add(_feat)

    # Build dynamic feature-abbreviation legend for caption
    _legend_parts: list[str] = []
    for _raw_set, _disp, _desc in _CAPTION_LEGEND:
        if used_raw_feats & _raw_set:
            _legend_parts.append(f"{_disp} = {_desc}")
    if has_sector_cluster:
        _legend_parts.append(r"sc:* = sector cluster membership")
    _feat_abbrev_caption = (
        r"Feature abbreviations: " + "; ".join(_legend_parts) + r"."
        if _legend_parts
        else ""
    )

    # -- Print LaTeX -----------------------------------------------------------
    _EVSET = eval_set_label
    _table_header = (
        r"\begin{table}[htbp]" + "\n"
        r"\caption{Outcome tag prediction results on the EVSET set. "
        r"$p < 0.05$ determined via permutation sampling on the EVSET set. "
        r"Consistent features shown are those in the top 10 by |SHAP|, "
        r"with the same sign, and at least 5\,\% of total importance, "
        r"in every split of 3 equal random training splits (RF and ET averaged). "
        r"Features are ordered by mean importance across splits; "
        r"features absent from any split's top 10 or exhibiting a sign flip are omitted. "
        r"$\overrightarrow{{\color[HTML]{006400}\text{Green}}}$ and "
        r"$\overleftarrow{{\color[HTML]{8B0000}\text{Red}}}$ indicate positive and negative "
        r"relationships respectively. "
        r"Tmp.\ Corr.: temporal correction in units of 1/100\,\%/yr. "
        + _feat_abbrev_caption
        + "\n"
        r"Groups and tags sorted by accuracy vs.\ baseline descending. }" + "\n"
        r"\label{tab:tag_model_results}" + "\n"
        r"\footnotesize" + "\n"
        r"\setlength{\tabcolsep}{3pt}" + "\n"
        r"\renewcommand{\arraystretch}{1.1}" + "\n"
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{2.87cm} r r r r r r r r r >{\raggedright\arraybackslash}X@{}}"
        + "\n"
        r"\toprule" + "\n"
        r"Tag (\* shown if $p<0.05$) & \makecell{WG\\Pair.\\Prob.} & \makecell{WG\\Spear.} & \makecell{Brier} & \makecell{Brier\\Base.} & Acc. & \makecell{Acc.\\Base.}"
        r" & \makecell{Frac.\ True\\(train,\ EVSET)}"
        r" & $N_\mathrm{EVSET}$"
        r" & \makecell{Tmp.\\Corr.} & \makecell{Consistent\\Features} \\" + "\n"
        r"\midrule"
    ).replace("EVSET", _EVSET)
    print(_table_header)

    skipped_tags: list[str] = []

    prev_group = None
    for gname, rows in groups:
        active_rows = [
            (d, t)
            for d, t in rows
            if results.get(t, {}).get("model_type", "const_base") != "const_base"
        ]
        skipped_rows = [
            (d, t)
            for d, t in rows
            if results.get(t, {}).get("model_type", "const_base") == "const_base"
        ]
        skipped_tags.extend(d for d, _ in skipped_rows)
        if not active_rows:
            continue

        if prev_group is not None:
            print(r"\noalign{\vspace{0.5ex}}")
        print(r"\multicolumn{11}{@{}l}{\textbf{" + gname + r"}} \\")
        prev_group = gname

        for display, tag in active_rows:
            r = results.get(tag, {})
            brier = r.get("val_brier", 0.0) or 0.0
            acc = r.get("val_acc", 0.0) or 0.0
            val_n_pos = r.get("val_n_pos") or 0
            val_n = r.get("val_n") or 1
            train_n_pos = r.get("train_n_pos") or 0
            train_n = r.get("train_n") or 1
            frac_true_val = val_n_pos / val_n
            frac_true_train = train_n_pos / train_n
            train_rate = train_n_pos / train_n if train_n > 0 else 0.5
            brier_base = r.get("val_brier_base") or (
                train_rate**2 * (1 - frac_true_val)
                + (1 - train_rate) ** 2 * frac_true_val
            )
            acc_base = max(train_rate, 1.0 - train_rate)

            consistent_feats = consistent_feats_cache.get(tag)
            feat_str = (
                _consistent_feat_cell(consistent_feats)
                if consistent_feats is not None
                else r"\textemdash"
            )

            slope = _year_correction_slope(
                tag, _year_corr_data, _yr_activity_ids, _yr_start_years
            )
            slope_str = f"{slope*100:+.2f}" if slope is not None else "None"

            pop = wg_pop_by_tag.get(tag)
            rank_str = f"{pop:.2f}" if pop is not None else r"\textemdash"

            wg_spear = wg_spearman_by_tag.get(tag)
            wg_spear_str = f"{wg_spear:.2f}" if wg_spear is not None else r"\textemdash"

            sig_marker = r"$^{\ast}$" if perm_p_by_tag.get(tag, 1.0) < 0.05 else ""

            print(
                f"{display}{sig_marker} & {rank_str} & {wg_spear_str} & {brier:.3f} & {brier_base:.3f} & {acc:.2f} & {acc_base:.2f}"
                f" & {frac_true_train:.2f},\\ {frac_true_val:.2f} & {val_n} & {slope_str} & {feat_str} \\\\"
            )

    print(r"""\bottomrule
\end{tabularx}
\end{table}""")

    if skipped_tags:
        print(
            f"The following tags fell back to const\\_base on the {eval_set_label} set: "
            + ", ".join(skipped_tags),
            file=sys.stderr,
        )

    # -- Summary metrics (WG-POP and Brier skill across curated 14 tags) -------
    curated_tags = [tag for _grp in TAG_GROUPS.values() for _lbl, tag in _grp]

    import numpy as _np

    rf_curated = [
        t
        for t in curated_tags
        if results.get(t, {}).get("model_type", "const_base") != "const_base"
    ]

    wg_pops = [
        wg_pop_by_tag[t]
        for t in rf_curated
        if t in wg_pop_by_tag and wg_pop_by_tag[t] is not None
    ]
    brier_skills = [
        results[t].get("val_brier_skill")
        for t in rf_curated
        if t in results and results[t].get("val_brier_skill") is not None
    ]

    wg_avg = int(round(_np.mean(wg_pops) * 100)) if wg_pops else 0
    wg_min = int(round(_np.min(wg_pops) * 100)) if wg_pops else 0
    wg_max = int(round(_np.max(wg_pops) * 100)) if wg_pops else 0
    bss_avg = float(_np.mean(brier_skills)) if brier_skills else 0.0

    print(
        f"\n% -- Summary metrics ({eval_set_label} set, {len(rf_curated)} RF+ET tags) --"
    )
    _pw_suffix = "Nolimits" if args.nolimits else "FeatSelJ"
    print(f"\\newcommand{{\\OutcomeTagPairwiseAvg{_pw_suffix}}}{{{wg_avg}}}")
    print(f"\\newcommand{{\\OutcomeTagPairwiseMin{_pw_suffix}}}{{{wg_min}}}")
    print(f"\\newcommand{{\\OutcomeTagPairwiseMax{_pw_suffix}}}{{{wg_max}}}")
    print(f"\\newcommand{{\\OutcomeTagBSS}}{{{bss_avg:.3f}}}")
    _n_sig = sum(1 for t in rf_curated if perm_p_by_tag.get(t, 1.0) < 0.05)
    print(f"TAGS_BETTER_THAN_CHANCE_95PCT: {_n_sig}")
    print(
        f"\n% Prose: Across {len(rf_curated)} binary outcome tags (covering activity rescoping, "
        f"finance and budget, and target achievement), the statistical model achieves an average "
        f"same-reporting-organisation, same-year pairwise ordering probability of "
        f"\\OutcomeTagPairwiseAvg\\,\\% "
        f"(range: \\OutcomeTagPairwiseMin\\,\\%--\\OutcomeTagPairwiseMax\\,\\%) "
        f"and an average Brier skill score of \\OutcomeTagBSS."
    )

    # -- Tag-vs-success correlation and cross-tag SHAP direction analysis -----
    if split_data:
        from feature_engineering import load_ratings as _load_ratings
        from scipy.stats import pearsonr as _pearsonr

        _ratings = _load_ratings(
            str(MERGED_OVERALL_RATINGS)
        )  # pd.Series indexed by activity_id

        _tags_df, _ = load_applied_tags(
            APPLIED_TAGS
        )  # DataFrame indexed by activity_id

        # Pearson correlation of each tag with overall rating (all activities)
        _tag_corrs: dict[str, float] = {}
        _all_ids = set(_ratings.index)
        for _stag in split_data:
            if _stag not in _tags_df.columns:
                continue
            _common = list(set(_tags_df[_stag].dropna().index) & _all_ids)
            if len(_common) < 20:
                continue
            _r, _ = _pearsonr(
                _tags_df.loc[_common, _stag].astype(float),
                _ratings.loc[_common],
            )
            _tag_corrs[_stag] = float(_r)

        print(
            "\n-- Tag correlation with six_overall_rating (all activities) --",
            file=sys.stderr,
        )
        for _stag, _corr in sorted(
            _tag_corrs.items(), key=lambda x: x[1], reverse=True
        ):
            _marker = " ***" if abs(_corr) > 0.05 else ""
            print(
                f"  {_stag.replace('tag_', ''):<55s}  r={_corr:+.4f}{_marker}",
                file=sys.stderr,
            )

        _qualifying = {t: c for t, c in _tag_corrs.items() if abs(c) > 0.05}
        print(
            f"\n-- {len(_qualifying)} tags with |r| > 0.05 --",
            file=sys.stderr,
        )
        for _stag, _corr in sorted(
            _qualifying.items(), key=lambda x: x[1], reverse=True
        ):
            _dir = "pos" if _corr > 0 else "neg"
            print(
                f"  {_stag.replace('tag_', ''):<55s}  r={_corr:+.4f}  ({_dir})",
                file=sys.stderr,
            )

        # For each qualifying tag: average signed SHAP across splits, flip if neg-correlated
        # Build feature -> list of sign-adjusted mean SHAP values (one per qualifying tag)
        _feat_adj: dict[str, list[float]] = {}
        for _stag, _corr in _qualifying.items():
            _td = split_data[_stag]
            _fcols = _td["feature_cols"]
            _avg_signed = _np.mean(
                [_np.asarray(s) for s in _td["split_signed_means"]], axis=0
            )
            _flip = -1.0 if _corr < 0 else 1.0
            for _fi, _fname in enumerate(_fcols):
                _val = float(_avg_signed[_fi]) * _flip
                _feat_adj.setdefault(_fname, []).append(_val)

        # Summarise each feature: fraction of qualifying tags where sign-adjusted SHAP > 0,
        # mean sign-adjusted SHAP, and how many qualifying tags include it.
        # Sort by fraction_positive desc, then mean_adj_shap desc.
        # Only show features that appear in >= MIN_TAGS qualifying tags.
        _MIN_TAGS = 2

        def _feat_summary(
            feat_adj: dict[str, list[float]], positive: bool
        ) -> list[tuple]:
            rows = []
            for _fname, _vals in feat_adj.items():
                if len(_vals) < _MIN_TAGS:
                    continue
                _n_agree = sum(1 for v in _vals if (v > 0) == positive)
                _frac = _n_agree / len(_vals)
                _mean = float(_np.mean(_vals))
                rows.append((_fname, _frac, _mean, len(_vals)))
            rows.sort(key=lambda x: (x[1], x[2] if positive else -x[2]), reverse=True)
            return rows

        print(
            f"\n-- Features associated with SUCCESS "
            f"(sign-adjusted SHAP, >={_MIN_TAGS} qualifying tags, sorted by agreement fraction) --",
            file=sys.stderr,
        )
        print(
            f"  {'feature':<45s}  {'frac_pos':>8s}  {'mean_adj':>9s}  {'n_tags':>6s}",
            file=sys.stderr,
        )
        for _fname, _frac, _mean, _n in _feat_summary(_feat_adj, positive=True):
            _bar = "#" * round(_frac * 10)
            print(
                f"  {_fname:<45s}  {_frac:>7.0%}  {_mean:>+9.4f}  {_n:>3d}/{len(_qualifying)}  {_bar}",
                file=sys.stderr,
            )

        print(
            f"\n-- Features associated with FAILURE "
            f"(sign-adjusted SHAP < 0, >={_MIN_TAGS} qualifying tags, sorted by agreement fraction) --",
            file=sys.stderr,
        )
        print(
            f"  {'feature':<45s}  {'frac_neg':>8s}  {'mean_adj':>9s}  {'n_tags':>6s}",
            file=sys.stderr,
        )
        for _fname, _frac, _mean, _n in _feat_summary(_feat_adj, positive=False):
            _bar = "#" * round(_frac * 10)
            print(
                f"  {_fname:<45s}  {_frac:>7.0%}  {_mean:>+9.4f}  {_n:>3d}/{len(_qualifying)}  {_bar}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    main()
