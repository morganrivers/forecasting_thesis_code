"""
Visualize tag model results.

Produces 4 plots saved to data/outcome_tags/tag_model_plots.png:
  1. Scatter: val POP vs val Accuracy, labeled with adjustText
  2. Bar: val POP sorted, colored by Brier skill
  3. Bar: Brier skill sorted by POP order
  4. Scatter: train minority class size vs val POP

Color scheme:
  green  (#2ca02c) -- Brier Skill >= 0.1
  orange (#ff7f0e) -- Brier Skill < 0.1, or val_acc below majority-class baseline
  grey   (#aaaaaa) -- const_base (no useful model found)

Shape scheme:
  o  circle   -- RF+ExtraTrees
  ^  triangle -- ridge logistic
  D  diamond  -- const_base (always grey)

Run after D_train_tag_predictors.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from adjustText import adjust_text as _adjust_text

SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from G_outcome_tag_train import (
    LATEST_TRAIN_POINT,
    LATEST_VALIDATION_POINT,
    TOO_LATE_CUTOFF,
)
from leakage_risk import EXCLUDE_TEST_LEAKAGE_RISK, TEST_LEAKAGE_RISK_IDS
from scoring_metrics import (
    adjusted_r2,
    within_group_pairwise_ordering_prob,
)

DATA_DIR = SCRIPT_DIR.parent.parent / "data"
OUT_DIR = DATA_DIR / "outcome_tags"
RESULTS_PATH = OUT_DIR / "tag_model_results.json"
MODELS_PATH = OUT_DIR / "tag_models.pkl"
OUT_PNG = OUT_DIR / "tag_model_plots.png"
OUT_CI_PNG = OUT_DIR / "tag_model_ci_plots.png"
OUT_WG_PNG = OUT_DIR / "tag_wg_pop_vs_train_size.png"

APPLIED_TAGS_PATH = OUT_DIR / "applied_tags.jsonl"
PREDICTIONS_PATH = OUT_DIR / "tag_predictions.csv"
INFO_CSV_PATH = DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"
MERGED_OVERALL_RATINGS = DATA_DIR / "merged_overall_ratings.jsonl"

EVAL_ON_TEST = False  # overridden by --test flag; uses test_idx instead of val_idx

TAG_GROUPS: dict[str, list[tuple[str, str]]] = {
    "Finance & budget": [
        ("Funds Cancelled or Unutilized", "tag_funds_cancelled_or_unutilized"),
        ("Funds Reallocated across components or activities.", "tag_funds_reallocated"),
        ("Full Disbursement of funds achieved", "tag_high_disbursement"),
        ("Over Budget", "tag_over_budget_success"),
    ],
    "Activity Rescoping": [
        ("Closing Date Extended", "tag_closing_date_extended"),
        ("Formally Restructured", "tag_project_restructured"),
        ("Targets Revised", "tag_targets_revised"),
    ],
    "Target achievement": [
        (
            "Targets Generally Met or Exceeded (% activities)",
            "tag_targets_met_or_exceeded_success",
        ),
        (
            "High Beneficiary Satisfaction or Reach (% activities)",
            "tag_high_beneficiary_satisfaction_or_reach_success",
        ),
        (
            "Private Sector Engagement (% activities)",
            "tag_private_sector_engagement_success",
        ),
        (
            "Capacity Building Delivered (% activities)",
            "tag_capacity_building_delivered_success",
        ),
        (
            "Gender-Equitable Outcomes (% activities)",
            "tag_gender_equitable_outcomes_success",
        ),
        (
            "Policy and Regulatory Reforms Achieved (% activities)",
            "tag_policy_regulatory_reforms_success_success",
        ),
        (
            "Improved Financial Performance (% activities)",
            "tag_improved_financial_performance",
        ),
    ],
}

EXCLUDE_ATTEMPTED_TAGS = True

POP_REF_HIGH = 0.70  # reference line only (no keep/drop logic)
POP_REF_LOW = 0.62  # reference line only

# -- Colour / shape helpers ----------------------------------------------------
C_GREEN = "#2ca02c"
C_ORANGE = "#ff7f0e"
C_GREY = "#aaaaaa"
C_RED_ANNOT = "#cc0000"  # connector lines in adjustText


def point_color(brier_skill, model_type: str, acc_below_baseline: bool = False) -> str:
    if model_type == "const_base":
        return C_GREY
    if acc_below_baseline or brier_skill is None or brier_skill < 0.1:
        return C_ORANGE
    return C_GREEN


def point_marker(model_type: str) -> str:
    return {
        "rf": "o",
        "rf+ET": "o",
        "rf+ridge": "s",
        "rf+ET+ridge": "s",
        "ridge": "^",
        "const_base": "D",
    }.get(model_type, "o")


def short_name(tag: str) -> str:
    return tag.replace("tag_", "").replace("_", " ")


def load_results(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def by_model_type(results: list[dict], model_type: str | tuple) -> dict[str, dict]:
    """Return {tag: result_dict} for a given model_type (or tuple of types)."""
    types = (model_type,) if isinstance(model_type, str) else model_type
    out = {}
    for r in results:
        if r.get("model_type") not in types:
            continue
        tag = r["tag"]
        if tag not in out or (r.get("val_pairwise_ordering_prob") or 0) > (
            out[tag].get("val_pairwise_ordering_prob") or 0
        ):
            out[tag] = r
    return out


def load_n_features() -> int | None:
    if not MODELS_PATH.exists():
        return None
    try:
        import pickle

        with MODELS_PATH.open("rb") as f:
            models_data = pickle.load(f)
        return len(models_data.get("feature_cols", []) or [])
    except Exception:
        return None


def _scatter_with_labels(ax, xs, ys, labels, colors, markers, sizes, edges, lws):
    """Plot scatter points and return adjustText Text objects."""
    texts = []
    for x, y, lbl, c, mk, s, ec, lw in zip(
        xs, ys, labels, colors, markers, sizes, edges, lws, strict=False
    ):
        ax.scatter(
            x,
            y,
            s=s,
            c=c,
            marker=mk,
            alpha=0.88,
            edgecolors=ec,
            linewidths=lw,
            zorder=3,
        )
        t = ax.text(x, y, lbl, fontsize=6.5, ha="left", va="bottom")
        texts.append(t)
    return texts


def _apply_adjusttext(ax, texts):
    _adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color=C_RED_ANNOT, lw=0.7),
        expand_points=(1.3, 2.0),
        expand_text=(1.2, 2.0),
        force_text=(0.4, 1.0),
        force_points=(0.2, 0.4),
    )


def _load_eval_data() -> tuple:
    """Load (preds_df, labels_df, eval_idx, group_key) for the current eval split.

    group_key is a Series of "reporting_org|||start_year" strings indexed by
    activity_id, restricted to eval_idx.

    Returns (None, None, None, None) if any required path is missing.
    """
    for path in (PREDICTIONS_PATH, APPLIED_TAGS_PATH, INFO_CSV_PATH):
        if not path.exists():
            return None, None, None, None

    preds_df = pd.read_csv(
        str(PREDICTIONS_PATH), index_col="activity_id", dtype={"activity_id": str}
    )

    info = pd.read_csv(
        str(INFO_CSV_PATH),
        usecols=[
            "activity_id",
            "txn_first_date",
            "actual_start_date",
            "original_planned_start_date",
            "reporting_orgs",
        ],
        dtype={"activity_id": str},
    )
    for col in ["txn_first_date", "actual_start_date", "original_planned_start_date"]:
        info[col] = pd.to_datetime(info[col], errors="coerce")
    info["start_date"] = (
        info["actual_start_date"]
        .combine_first(info["original_planned_start_date"])
        .combine_first(info["txn_first_date"])
    )
    info["start_year"] = info["start_date"].dt.year
    info = info.set_index("activity_id").reindex(preds_df.index)

    too_late = info[info["start_date"] >= pd.to_datetime(TOO_LATE_CUTOFF)].index
    remaining = info.drop(index=too_late, errors="ignore")
    test_idx = remaining[
        remaining["start_date"] > pd.to_datetime(LATEST_VALIDATION_POINT)
    ].index
    if EXCLUDE_TEST_LEAKAGE_RISK:
        test_idx = test_idx.difference(pd.Index(list(TEST_LEAKAGE_RISK_IDS)))
    remaining2 = remaining.drop(index=test_idx, errors="ignore")
    val_idx = remaining2[
        remaining2["start_date"] > pd.to_datetime(LATEST_TRAIN_POINT)
    ].index
    eval_idx = test_idx if EVAL_ON_TEST else val_idx

    labels_df = _load_all_applied_tags()
    unsigned_cols = [c for c in labels_df.columns if not c.endswith("_success")]
    labels_df[unsigned_cols] = labels_df[unsigned_cols].fillna(0)

    group_key = (
        info.reindex(eval_idx)["reporting_orgs"].fillna("unknown").astype(str)
        + "|||"
        + info.reindex(eval_idx)["start_year"].fillna(-1).astype(int).astype(str)
    )

    return preds_df, labels_df, eval_idx, group_key


def compute_raw_brier_scores(
    exclude_attempted: bool = True,
) -> dict[str, dict[str, float]]:
    """Return raw Brier score (mean squared error) per tag on the eval set.

    Returns {tag: {"rf": model_brier, "const_base": baseline_brier}}.
    Reads val_y_true / val_y_pred / val_brier_base saved by G_outcome_tag_train.
    Use brier_for_model_type() to look up the right score for a given model type.
    """
    if not RESULTS_PATH.exists():
        return {}
    results = load_results(RESULTS_PATH)
    out: dict[str, dict[str, float]] = {}
    for r in results:
        tag = r.get("tag", "")
        if exclude_attempted and "attempted" in tag:
            continue
        y_true_list = r.get("val_y_true")
        y_pred_list = r.get("val_y_pred")
        if not y_true_list or not y_pred_list:
            continue
        y_true = np.array(y_true_list, dtype=float)
        y_pred = np.array(y_pred_list, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() == 0:
            continue
        model_brier = float(np.mean((y_pred[mask] - y_true[mask]) ** 2))
        mtype = r.get("model_type", "rf")
        mkey = "const_base" if mtype == "const_base" else "rf"
        out.setdefault(tag, {})[mkey] = model_brier
        # Always store baseline Brier so brier_for_model_type("const_base") works for any tag
        brier_base = r.get("val_brier_base")
        if brier_base is not None:
            out[tag]["const_base"] = float(brier_base)
    return out


def brier_for_model_type(
    raw_brier_by_tag: dict[str, dict[str, float]], tag: str, model_type: str
) -> float | None:
    """Return the raw Brier score for the *chosen* model type for a tag."""
    per_tag = raw_brier_by_tag.get(tag)
    if per_tag is None:
        return None
    if model_type == "const_base":
        return per_tag.get("const_base")
    if model_type == "ridge":
        return per_tag.get("ridge")
    # rf, rf+ET, rf+ridge, rf+ET+ridge -- all stored under "rf"
    return per_tag.get("rf")


def compute_wg_pop_with_group_stats(exclude_attempted: bool = True) -> dict[str, dict]:
    """
    Returns per-tag within-group POP plus group statistics:
    wg_pop, n_years, n_orgs, min/median/max pairs per (org, year) group on the val set.
    """
    preds_df, labels_df, eval_idx, group_key = _load_eval_data()
    if preds_df is None:
        return {}

    labels_val = labels_df.reindex(eval_idx)

    result: dict[str, dict] = {}
    for pred_col in preds_df.columns:
        for suffix in ("__rf", "__ridge", "__const_base"):
            if pred_col.endswith(suffix):
                tag = pred_col[: -len(suffix)]
                break
        else:
            continue
        if exclude_attempted and "attempted" in tag:
            continue
        if tag not in labels_val.columns:
            continue

        y_true_s = labels_val[tag]
        y_pred_s = preds_df.loc[eval_idx, pred_col]

        y_true_arr = y_true_s.reindex(group_key.index).to_numpy(dtype=float)
        y_pred_arr = y_pred_s.reindex(group_key.index).to_numpy(dtype=float)
        grp_arr = group_key.to_numpy(dtype=str)

        try:
            wg_pop_val = within_group_pairwise_ordering_prob(
                y_true_arr, y_pred_arr, grp_arr
            )["prob"]
            wg_pop_val = float(wg_pop_val) if np.isfinite(wg_pop_val) else None
        except Exception:
            wg_pop_val = None

        # Count differing-outcome pairs (y_true[i] != y_true[j]) within groups
        n_differing = 0
        n_total_pairs = 0
        mask_diff = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        for g in np.unique(grp_arr[mask_diff]):
            gmask = mask_diff & (grp_arr == g)
            yt_g = y_true_arr[gmask]
            if len(yt_g) < 2:
                continue
            i_idx, j_idx = np.triu_indices(len(yt_g), k=1)
            dt = yt_g[i_idx] - yt_g[j_idx]
            n_total_pairs += len(dt)
            n_differing += int(np.sum(dt != 0))

        # Group stats: activities per (org, year) group with valid y_true + y_pred
        mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        valid_grps = grp_arr[mask]
        if len(valid_grps) == 0:
            result[tag] = {
                "wg_pop": wg_pop_val,
                "n_years": 0,
                "n_orgs": 0,
                "min_pairs": 0,
                "med_pairs": 0,
                "max_pairs": 0,
                "n_differing_pairs": n_differing,
                "n_total_pairs": n_total_pairs,
            }
            continue

        unique_grps, counts = np.unique(valid_grps, return_counts=True)
        active_mask = counts >= 2
        active_grps = unique_grps[active_mask]
        active_counts = counts[active_mask]

        if len(active_grps) == 0:
            result[tag] = {
                "wg_pop": wg_pop_val,
                "n_years": 0,
                "n_orgs": 0,
                "min_pairs": 0,
                "med_pairs": 0,
                "max_pairs": 0,
                "n_differing_pairs": n_differing,
                "n_total_pairs": n_total_pairs,
            }
            continue

        pairs_per_grp = active_counts * (active_counts - 1) // 2
        years: set[str] = set()
        orgs: set[str] = set()
        for g in active_grps:
            parts = g.split("|||")
            orgs.add(parts[0])
            if len(parts) > 1:
                years.add(parts[1])

        result[tag] = {
            "wg_pop": wg_pop_val,
            "n_years": len(years),
            "n_orgs": len(orgs),
            "min_pairs": int(pairs_per_grp.min()),
            "med_pairs": int(np.median(pairs_per_grp)),
            "max_pairs": int(pairs_per_grp.max()),
            "n_differing_pairs": n_differing,
            "n_total_pairs": n_total_pairs,
        }

    return result


def _load_all_applied_tags() -> pd.DataFrame:
    """Load all applied_tags.jsonl rows into a DataFrame indexed by activity_id."""
    records = []
    with APPLIED_TAGS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    label_rows: dict[str, dict] = {}
    for obj in records:
        aid = str(obj.get("activity_id", ""))
        row: dict = {}
        for tag, present in obj.get("unsigned_tags", {}).items():
            row[f"tag_{tag}"] = int(bool(present))
        for tag, success in obj.get("signed_tags", {}).items():
            row[f"tag_{tag}_success"] = (
                np.nan if success is None else int(bool(success))
            )
        label_rows[aid] = row
    return pd.DataFrame.from_dict(label_rows, orient="index")


def print_grouped_tag_summary(summary_by_pop: list[dict]) -> None:
    """Print categorised tag list with prevalence, Pearson correlation, and val accuracy."""
    if not APPLIED_TAGS_PATH.exists():
        print("WARNING: applied_tags.jsonl not found -- skipping grouped summary.")
        return

    labels_df = _load_all_applied_tags()
    total_n = len(labels_df)

    # Load ratings for Pearson correlation (graceful fallback)
    ratings: pd.Series | None = None
    try:
        from feature_engineering import load_ratings

        if MERGED_OVERALL_RATINGS.exists():
            ratings = load_ratings(str(MERGED_OVERALL_RATINGS))
    except Exception:
        pass

    raw_brier_by_tag = compute_raw_brier_scores(
        exclude_attempted=EXCLUDE_ATTEMPTED_TAGS
    )

    acc_by_tag = {r["tag"]: r.get("rf_acc") for r in summary_by_pop}
    mtype_by_tag = {r["tag"]: r.get("model_type", "rf+ET") for r in summary_by_pop}
    info_by_tag = {r["tag"]: r for r in summary_by_pop}

    def _baseline_acc_for(tag_col: str) -> float | None:
        r = info_by_tag.get(tag_col)
        if r is None:
            return None
        tr_pos = r.get("train_n_pos") or 0
        tr_n = r.get("train_n") or 0
        if tr_n == 0:
            return None
        tr_rate = tr_pos / tr_n
        val_n = r.get("val_n") or 0
        val_n_pos = r.get("val_n_pos") or 0
        if val_n == 0:
            return None
        val_pos_rate = val_n_pos / val_n
        return val_pos_rate if tr_rate >= 0.5 else (1.0 - val_pos_rate)

    print(f"\n{'='*85}")
    print(f"OUTCOME TAG SUMMARY  --  Total activities with tags applied: {total_n}")
    print(
        f"Activity %: percent of {total_n:,} applicable activities containing the tag"
    )
    print("Ratio: Accuracy ratio to baseline of mean prediction")
    print(f"{'='*85}")

    all_grouped_tags: set[str] = set()

    for group_name, tag_items in TAG_GROUPS.items():
        print(f"\n{group_name}")
        for display, tag_col in tag_items:
            all_grouped_tags.add(tag_col)

            col_data = (
                labels_df[tag_col].dropna()
                if tag_col in labels_df.columns
                else pd.Series(dtype=float)
            )
            pct_str = f"{round(col_data.mean()*100):d}%" if len(col_data) > 0 else "N/A"

            # Pearson correlation with overall rating (as integer percent)
            corr_str = "N/A"
            if ratings is not None and tag_col in labels_df.columns:
                common = (
                    labels_df[tag_col]
                    .dropna()
                    .index.intersection(ratings.dropna().index)
                )
                if len(common) > 5:
                    r_val = float(
                        np.corrcoef(
                            labels_df.loc[common, tag_col].astype(float),
                            ratings.loc[common].astype(float),
                        )[0, 1]
                    )
                    sign = "+" if r_val >= 0 else ""
                    corr_str = f"{sign}{round(r_val*100):d}%"

            acc = acc_by_tag.get(tag_col)
            bl = _baseline_acc_for(tag_col)
            ratio_str = f"{acc/bl:.2f}" if (acc is not None and bl) else "N/A"
            acc_str = f"{round(acc*100):d}%" if acc is not None else "N/A"
            raw_brier = brier_for_model_type(
                raw_brier_by_tag, tag_col, mtype_by_tag.get(tag_col, "rf+ET")
            )
            brier_str = f"{raw_brier:.3f}" if raw_brier is not None else "N/A"

            print(
                f" - {display}, Activity %: {pct_str}, Ratio: {ratio_str}, Pearson correlation: {corr_str}, Accuracy: {acc_str}, Brier: {brier_str}"
            )

    # Tags extracted but const_base (no useful model found), not in the grouped list
    not_predicted = [
        r
        for r in summary_by_pop
        if r["model_type"] == "const_base" and r["tag"] not in all_grouped_tags
    ]
    if not_predicted:
        print("\nExtracted but not able to be predicted:")
        for r in not_predicted:
            name = short_name(r["tag"])
            print(f" - {name[0].upper()}{name[1:]}")

    print()


def plot_wg_pop_figure(summary_by_pop: list[dict], wg_stats: dict[str, dict]) -> None:
    """Separate PNG: within-group POP vs train minority class size, labeled with group stats."""
    eval_label = "test" if EVAL_ON_TEST else "val"
    fig, ax = plt.subplots(figsize=(11, 8))

    legend_color_patches = [
        mpatches.Patch(color=C_GREEN, label="Brier Skill >= 0.1"),
        mpatches.Patch(color=C_ORANGE, label="Brier Skill < 0.1 or acc < baseline"),
    ]
    legend_shape_proxies = [
        plt.scatter([], [], s=70, c="grey", marker="o", label="RF+ExtraTrees"),
        plt.scatter([], [], s=70, c="grey", marker="^", label="Ridge Regression"),
    ]

    xs, ys, labels, colors, markers, sizes, edges, lws = [], [], [], [], [], [], [], []
    for row in summary_by_pop:
        tag = row["tag"]
        if row["model_type"] == "const_base":
            continue
        stats = wg_stats.get(tag, {})
        wg_pop_val = stats.get("wg_pop")
        if wg_pop_val is None:
            continue
        tr_pos = row.get("train_n_pos") or 0
        tr_n = row.get("train_n") or 0
        minority = min(tr_pos, tr_n - tr_pos)
        if minority == 0:
            continue

        mtype = row["model_type"]
        c = point_color(row["rf_brier"], mtype, row.get("acc_below_baseline", False))
        mk = point_marker(mtype)
        ec = "white"
        lw = 0.5

        n_yr = stats.get("n_years", 0)
        n_org = stats.get("n_orgs", 0)
        min_p = stats.get("min_pairs", 0)
        med_p = stats.get("med_pairs", 0)
        max_p = stats.get("max_pairs", 0)
        lbl = f"{row['short']}\n{n_yr}yr {n_org}org {min_p}/{med_p}/{max_p}pr"

        xs.append(minority)
        ys.append(wg_pop_val)
        labels.append(lbl)
        colors.append(c)
        markers.append(mk)
        sizes.append(80)
        edges.append(ec)
        lws.append(lw)

    texts = _scatter_with_labels(ax, xs, ys, labels, colors, markers, sizes, edges, lws)

    ax.axhline(0.5, color="grey", linestyle="-", linewidth=0.7)
    ax.axhline(
        POP_REF_HIGH,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=f"POP={POP_REF_HIGH}",
    )
    ax.axhline(
        POP_REF_LOW,
        color="grey",
        linestyle=":",
        linewidth=0.8,
        label=f"POP={POP_REF_LOW}",
    )

    ax.set_xlabel("Train minority class count  [min(N_pos, N_neg)]", fontsize=10)
    ax.set_ylabel(f"Within-group POP ({eval_label})", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_title(
        "Within-group POP vs train minority class size\n"
        "Label: tag / #years #orgs min/med/max pairs per (org, year) group\n"
        "(colour = Brier Skill; grey = constant baseline)",
        fontsize=10,
    )
    reflines, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=legend_color_patches + legend_shape_proxies + reflines,
        fontsize=7.5,
        loc="upper left",
    )
    ax.grid(True, alpha=0.3, zorder=0)
    _apply_adjusttext(ax, texts)

    plt.tight_layout()
    plt.savefig(OUT_WG_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_WG_PNG}")


def _acc_below_baseline(src: dict) -> bool:
    """True if val_acc is below the majority-class baseline (predict train majority on val)."""
    acc = src.get("val_acc")
    tr_n = src.get("train_n") or 0
    tr_pos = src.get("train_n_pos") or 0
    val_n = src.get("val_n") or 0
    val_pos = src.get("val_n_pos") or 0
    if acc is None or tr_n == 0 or val_n == 0:
        return False
    tr_rate = tr_pos / tr_n
    val_pos_rate = val_pos / val_n
    baseline = val_pos_rate if tr_rate >= 0.5 else (1.0 - val_pos_rate)
    return acc < baseline


def plot_from_results(
    results: list[dict],
    skip_ci: bool = True,
    baseline_results: list[dict] | None = None,
) -> None:
    """Core plotting logic. Accepts results as a list[dict] directly.

    This is the same logic as ``main()`` but decoupled from file I/O so that
    callers (e.g. G_outcome_tag_train.py) can pass in-memory results without first
    writing them to disk.
    """
    n_features = load_n_features()
    eval_label = "test" if EVAL_ON_TEST else "val"
    raw_brier_by_tag = compute_raw_brier_scores(
        exclude_attempted=EXCLUDE_ATTEMPTED_TAGS
    )
    rf_by_tag = by_model_type(results, ("rf", "rf+ET", "rf+ridge", "rf+ET+ridge"))
    ridge_by_tag = by_model_type(results, "ridge")
    const_base_by_tag = by_model_type(results, "const_base")
    all_tags = sorted(set(rf_by_tag) | set(ridge_by_tag) | set(const_base_by_tag))

    if EXCLUDE_ATTEMPTED_TAGS:
        all_tags = [t for t in all_tags if "attempted" not in t]

    wg_stats = compute_wg_pop_with_group_stats(exclude_attempted=EXCLUDE_ATTEMPTED_TAGS)

    summary = []
    for tag in all_tags:
        # Use the chosen model (D_ saves exactly one result per tag)
        if tag in rf_by_tag:
            src = rf_by_tag[tag]
            mtype = src.get("model_type", "rf")
        elif tag in ridge_by_tag:
            src = ridge_by_tag[tag]
            mtype = "ridge"
        else:
            src = const_base_by_tag.get(tag, {})
            mtype = "const_base"

        n_tr = src.get("train_n", 0) or 0
        tr_brier = src.get("train_brier_skill")
        tr_pop = src.get("train_pairwise_ordering_prob")
        vl_brier = src.get("val_brier_skill")
        adj_r2 = adjusted_r2(tr_brier, n_tr, n_features) if n_features else np.nan
        overfit = (
            (adj_r2 - vl_brier)
            if (np.isfinite(adj_r2) and vl_brier is not None)
            else None
        )
        vl_pop = src.get("val_pairwise_ordering_prob")
        pop_gap = (
            (tr_pop - vl_pop) if (tr_pop is not None and vl_pop is not None) else None
        )

        summary.append(
            {
                "tag": tag,
                "short": short_name(tag),
                "model_type": mtype,
                "train_n_pos": src.get("train_n_pos", 0),
                "train_n": n_tr,
                "val_n_pos": src.get("val_n_pos", 0),
                "val_n": src.get("val_n", 0),
                "rf_pop": src.get("val_pairwise_ordering_prob"),
                "within_grp_pop": (wg_stats.get(tag) or {}).get("wg_pop"),
                "rf_acc": src.get("val_acc"),
                "rf_brier": vl_brier,
                "track": src.get("track"),
                "train_base_rate": src.get("train_base_rate"),
                "train_pop": tr_pop,
                "train_brier": tr_brier,
                "adj_r2_train": adj_r2,
                "overfit_gap": overfit,
                "pop_gap": pop_gap if mtype != "const_base" else None,
                "is_summary": not (
                    tag.endswith("_attempted") or tag.endswith("_success")
                ),
                "acc_below_baseline": _acc_below_baseline(src),
            }
        )

    summary_by_pop = sorted(summary, key=lambda r: r["rf_pop"] or 0, reverse=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 26))
    fig.suptitle(
        "Tag Model Results -- primary metric: POP (pairwise ordering probability)",
        fontsize=13,
        fontweight="bold",
        y=0.998,
    )

    # -- Legend proxies ---------------------------------------------------------
    def _legend_color_patches():
        return [
            mpatches.Patch(color=C_GREEN, label="Brier Skill >= 0.1, acc >= baseline"),
            mpatches.Patch(color=C_ORANGE, label="Brier Skill < 0.1 or acc < baseline"),
            mpatches.Patch(color=C_GREY, label="Constant Baseline"),
        ]

    def _legend_shape_proxies():
        return [
            plt.scatter([], [], s=70, c="grey", marker="o", label="RF+ExtraTrees"),
            plt.scatter([], [], s=70, c="grey", marker="^", label="Ridge Regression"),
            plt.scatter(
                [],
                [],
                s=60,
                c=C_GREY,
                marker="D",
                edgecolors="darkgrey",
                linewidths=0.8,
                label="Constant Baseline",
            ),
        ]

    # -- Plot 1: Scatter POP vs Accuracy ---------------------------------------
    ax = axes[0]
    xs, ys, labels, colors, markers, sizes, edges, lws = [], [], [], [], [], [], [], []
    for row in summary:
        if row["rf_pop"] is None or row["rf_acc"] is None:
            continue
        mtype = row["model_type"]
        c = point_color(row["rf_brier"], mtype, row.get("acc_below_baseline", False))
        mk = point_marker(mtype)
        s = 60 if mtype == "const_base" else 90
        ec = "darkgrey" if mtype == "const_base" else "white"
        lw = 0.8 if mtype == "const_base" else 0.5
        xs.append(row["rf_acc"])
        ys.append(row["rf_pop"])
        labels.append(row["short"])
        colors.append(c)
        markers.append(mk)
        sizes.append(s)
        edges.append(ec)
        lws.append(lw)

    texts1 = _scatter_with_labels(
        ax, xs, ys, labels, colors, markers, sizes, edges, lws
    )

    ax.axhline(POP_REF_HIGH, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(POP_REF_LOW, color="grey", linestyle=":", linewidth=0.8)
    ax.axvline(0.75, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"{eval_label.capitalize()} Accuracy", fontsize=10)
    ax.set_ylabel(f"{eval_label.capitalize()} POP", fontsize=10)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 0.95)
    ax.set_title(
        "1. POP vs Accuracy -- each point is one tag\n"
        "(colour = Brier Skill level; shape = model type: o RF+ET, ^ Ridge, * Constant Baseline)",
        fontsize=10,
    )
    ax.legend(
        handles=_legend_color_patches()
        + _legend_shape_proxies()
        + [
            plt.Line2D(
                [0], [0], color="black", linestyle="--", label=f"POP={POP_REF_HIGH}"
            ),
            plt.Line2D(
                [0], [0], color="grey", linestyle=":", label=f"POP={POP_REF_LOW}"
            ),
        ],
        fontsize=7.5,
        loc="lower right",
    )
    ax.grid(True, alpha=0.3, zorder=0)
    _apply_adjusttext(ax, texts1)

    # -- Plot 2: Bar -- val POP sorted -------------------------------------------
    ax = axes[1]
    n = len(summary_by_pop)
    x = np.arange(n)
    tags_pop = [r["short"] for r in summary_by_pop]
    pops = [r["rf_pop"] or 0 for r in summary_by_pop]
    bar_colors = [
        point_color(r["rf_brier"], r["model_type"], r.get("acc_below_baseline", False))
        for r in summary_by_pop
    ]
    hatches = ["//" if r["model_type"] == "ridge" else "" for r in summary_by_pop]

    # Accuracy ratio = val_acc / baseline_acc
    # baseline_acc = majority-class prediction accuracy on val set using training base rate
    def _baseline_acc(r):
        tr_pos = r.get("train_n_pos") or 0
        tr_n = r.get("train_n") or 1
        val_n = r.get("val_n") or 0
        val_n_pos = r.get("val_n_pos") or 0
        tr_rate = tr_pos / tr_n
        val_pos_rate = val_n_pos / val_n if val_n > 0 else tr_rate
        return val_pos_rate if tr_rate >= 0.5 else (1.0 - val_pos_rate)

    acc_ratios = []
    for r in summary_by_pop:
        bl = _baseline_acc(r)
        ac = r.get("rf_acc") or 0
        acc_ratios.append(ac / bl if bl > 0 else 1.0)

    # Right y-axis: scale so ratio=1.0 aligns with POP=0.5 on left axis.
    # Left ylim=(0.45, 0.95): POP=0.5 is at fraction 0.10 from bottom.
    # Match that fraction on right axis with a 0.50-unit range (same as left).
    LEFT_BOT, LEFT_TOP = 0.45, 0.95
    RIGHT_RANGE = LEFT_TOP - LEFT_BOT  # keep same visual range
    frac_baseline = (0.5 - LEFT_BOT) / (LEFT_TOP - LEFT_BOT)  # = 0.10
    right_bot = 1.0 - frac_baseline * RIGHT_RANGE  # = 0.95
    right_top = right_bot + RIGHT_RANGE  # = 1.45

    bw = 0.35
    bars_pop = ax.bar(x - bw / 2, pops, bw, color=bar_colors, alpha=0.9, label="POP")
    for bar, hatch in zip(bars_pop, hatches, strict=False):
        bar.set_hatch(hatch)

    ax2 = ax.twinx()
    bars_acc = ax2.bar(
        x + bw / 2,
        acc_ratios,
        bw,
        color=bar_colors,
        alpha=0.55,
        edgecolor="black",
        linewidth=0.4,
        label="Acc ratio",
    )
    for bar, hatch in zip(bars_acc, hatches, strict=False):
        bar.set_hatch(hatch)
    ax2.set_ylim(right_bot, right_top)
    ax2.set_ylabel("Accuracy / baseline accuracy", fontsize=10)
    ax2.axhline(1.0, color="darkgrey", linestyle="-", linewidth=0.7)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}x")
    )

    ax.axhline(POP_REF_HIGH, color="black", linestyle="--", linewidth=0.9)
    ax.axhline(POP_REF_LOW, color="grey", linestyle=":", linewidth=0.9)
    ax.axhline(0.5, color="darkgrey", linestyle="-", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tags_pop, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel(f"{eval_label.capitalize()} POP", fontsize=10)
    ax.set_ylim(LEFT_BOT, LEFT_TOP)
    ax.set_title(
        f"2. {eval_label.capitalize()} POP (left) and accuracy / baseline accuracy (right, semi-transparent) per tag\n"
        "(green = Brier Skill >= 0.1, orange = Brier Skill < 0.1, grey = Constant Baseline; hatch = Ridge)",
        fontsize=10,
    )
    ridge_patch = mpatches.Patch(
        facecolor="white", edgecolor="black", hatch="//", label="Ridge"
    )
    ax.legend(
        handles=_legend_color_patches()
        + [ridge_patch]
        + [
            plt.Line2D(
                [0], [0], color="black", linestyle="--", label=f"POP={POP_REF_HIGH}"
            ),
            plt.Line2D(
                [0], [0], color="grey", linestyle=":", label=f"POP={POP_REF_LOW}"
            ),
            plt.Line2D(
                [0], [0], color="darkgrey", linestyle="-", label="POP=0.5 / ratio=1.0"
            ),
        ],
        fontsize=8,
    )
    ax.grid(True, axis="y", alpha=0.3)

    # -- Plot 3: Bar -- Brier skill -----------------------------------------------
    ax = axes[2]
    briers = [r["rf_brier"] or 0 for r in summary_by_pop]
    bar_colors3 = [
        point_color(r["rf_brier"], r["model_type"], r.get("acc_below_baseline", False))
        for r in summary_by_pop
    ]

    bars3 = ax.bar(x, briers, 0.7, color=bar_colors3, alpha=0.9)
    for bar, hatch in zip(bars3, hatches, strict=False):
        bar.set_hatch(hatch)

    ax.axhline(0, color="black", linewidth=1.2, label="Brier skill = 0 (baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(tags_pop, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Brier Skill Score", fontsize=10)
    ax.set_title(
        "3. Brier Skill Score (same tag order as plot 2 -- sorted by POP)\n"
        "(positive = better than predicting base rate; hatch = ridge)",
        fontsize=10,
    )
    ax.legend(
        handles=_legend_color_patches()
        + [ridge_patch]
        + [
            plt.Line2D(
                [0], [0], color="black", linewidth=1.2, label="Brier = 0 (baseline)"
            ),
        ],
        fontsize=8,
    )
    ax.grid(True, axis="y", alpha=0.3)

    # -- Plot 4: Scatter train minority N vs POP --------------------------------
    ax = axes[3]
    xs4, ys4, labels4, colors4, markers4, sizes4, edges4, lws4 = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for row in summary:
        if not row["train_n_pos"] or row["rf_pop"] is None:
            continue
        mtype = row["model_type"]
        c = point_color(row["rf_brier"], mtype, row.get("acc_below_baseline", False))
        mk = point_marker(mtype)
        bs = row["rf_brier"] or 0
        s = max(40, min(320, abs(bs) * 600 + 50)) if mtype != "const_base" else 60
        ec = "darkgrey" if mtype == "const_base" else "white"
        lw = 0.8 if mtype == "const_base" else 0.5
        tr_pos = row.get("train_n_pos") or 0
        tr_neg = (row.get("train_n") or 0) - tr_pos
        minority = min(tr_pos, tr_neg)
        xs4.append(minority)
        ys4.append(row["rf_acc"] or 0)
        labels4.append(row["short"])
        colors4.append(c)
        markers4.append(mk)
        sizes4.append(s)
        edges4.append(ec)
        lws4.append(lw)

    texts4 = _scatter_with_labels(
        ax, xs4, ys4, labels4, colors4, markers4, sizes4, edges4, lws4
    )

    ax.set_xlabel("Train minority class count  [min(N_pos, N_neg)]", fontsize=10)
    ax.set_ylabel(f"{eval_label.capitalize()} Accuracy", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(0.45, 0.95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title(
        f"4. Minority class size vs {eval_label.capitalize()} Accuracy\n"
        "(bubble size = |Brier skill|; colour = Brier Skill level; shape = model type)",
        fontsize=10,
    )
    ax.legend(
        handles=_legend_color_patches() + _legend_shape_proxies(),
        fontsize=7.5,
        loc="upper left",
    )
    ax.grid(True, alpha=0.3, zorder=0)
    _apply_adjusttext(ax, texts4)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")

    plot_wg_pop_figure(summary_by_pop, wg_stats)

    print_grouped_tag_summary(summary_by_pop)

    # -- Console summary table --------------------------------------------------
    other = [r for r in summary_by_pop if "_attempted" not in r["tag"]]

    # Build baseline POP lookup from baseline_results (best POP per tag)
    base_pop_by_tag: dict[str, float] = {}
    if baseline_results:
        for r in baseline_results:
            tag = r["tag"]
            pop = r.get("val_pairwise_ordering_prob")
            if pop is not None and pop > base_pop_by_tag.get(tag, -1):
                base_pop_by_tag[tag] = pop

    show_baseline = bool(base_pop_by_tag)

    print(
        "\n=== Tag model summary (sorted by POP) -- RF+ET rows; [CB] = const_base fallback ==="
    )
    hdr = (
        f"{'model':10s}  {'kind':4s}  {'tag':52s}  {'val_pop':>7s}  {'wg_pop':>7s}"
        + (f"  {'base':>7s}  {'Deltapop':>6s}" if show_baseline else "")
        + f"  {'val_brier':>9s}"
        f"  {'tr_pop':>6s}  {'tr_brier':>8s}  {'adj_R2':>7s}  {'overfit':>7s}  {'popgap':>7s}"
        f"  {'tr_pos':>6s}  {'tr_neg':>6s}  {'vl_pos':>5s}  {'vl_n':>5s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in other:
        is_cb = row["model_type"] == "const_base"
        kind = (
            ("[SCB]" if row["is_summary"] else "[CB] ")
            if is_cb
            else ("[S]  " if row["is_summary"] else "     ")
        )
        pop_s = f"{row['rf_pop']:.3f}" if row["rf_pop"] is not None else "    N/A"
        wg_s = (
            f"{row['within_grp_pop']:.3f}"
            if row["within_grp_pop"] is not None
            else "    N/A"
        )
        brier_s = (
            f"{row['rf_brier']:+.3f}" if row["rf_brier"] is not None else "     N/A"
        )
        if is_cb:
            tr_pop_s = "    ---"
            tr_br_s = "     ---"
            adj_s = "    ---"
            ovf_s = "    ---"
            pop_gap_s = "    ---"
        else:
            tr_pop_s = (
                f"{row['train_pop']:.3f}" if row["train_pop"] is not None else "    N/A"
            )
            tr_br_s = (
                f"{row['train_brier']:+.3f}"
                if row["train_brier"] is not None
                else "     N/A"
            )
            adj_s = (
                f"{row['adj_r2_train']:+.3f}"
                if row["adj_r2_train"] is not None
                else "    N/A"
            )
            ovf_s = (
                f"{row['overfit_gap']:+.3f}"
                if row["overfit_gap"] is not None
                else "    N/A"
            )
            pop_gap_s = (
                f"{row['pop_gap']:+.3f}" if row["pop_gap"] is not None else "    N/A"
            )
        tr_neg = (row["train_n"] or 0) - (row["train_n_pos"] or 0)
        baseline_cols = ""
        if show_baseline:
            base = base_pop_by_tag.get(row["tag"])
            base_s = f"{base:.3f}" if base is not None else "    N/A"
            delta = (
                (row["rf_pop"] - base)
                if (row["rf_pop"] is not None and base is not None)
                else None
            )
            delta_s = f"{delta:+.3f}" if delta is not None else "    N/A"
            baseline_cols = f"  {base_s:>7s}  {delta_s:>6s}"
        print(
            f"{row['model_type']:10s}  {kind}  {row['tag']:52s}  {pop_s:>7s}  {wg_s:>7s}"
            + baseline_cols
            + f"  {brier_s:>9s}"
            f"  {tr_pop_s:>6s}  {tr_br_s:>8s}  {adj_s:>7s}  {ovf_s:>7s}  {pop_gap_s:>7s}"
            f"  {row['train_n_pos']:>6d}  {tr_neg:>6d}  {row['val_n_pos']:>5d}  {row['val_n']:>5d}"
        )

    n_rf = sum(1 for r in summary if r["model_type"] == "rf")
    n_rfet = sum(1 for r in summary if r["model_type"] == "rf+ET")
    n_rfridge = sum(1 for r in summary if r["model_type"] == "rf+ridge")
    n_ridge = sum(1 for r in summary if r["model_type"] == "ridge")
    n_cb = sum(1 for r in summary if r["model_type"] == "const_base")
    print(
        f"\nModel types chosen -- rf: {n_rf}  rf+ET: {n_rfet}  rf+ridge: {n_rfridge}  ridge: {n_ridge}  const_base: {n_cb}"
    )

    print(f"\n{'-'*52}")
    print(f"{'metric':8s}  {'mean':>7s}  {'median':>7s}  {'peak':>7s}  {'min':>7s}  n")
    print(f"{'-'*52}")
    for label, key, fmt in [
        ("pop", "rf_pop", ".3f"),
        ("acc", "rf_acc", ".1%"),
        ("brier", "rf_brier", "+.3f"),
    ]:
        vals = [r[key] for r in summary if r[key] is not None]
        if not vals:
            continue

        def _f(v, f=fmt):
            return f"{v:{f}}"

        print(
            f"{label:8s}  {_f(np.mean(vals)):>7s}  {_f(np.median(vals)):>7s}"
            f"  {_f(np.max(vals)):>7s}  {_f(np.min(vals)):>7s}  {len(vals)}"
        )
    print(f"{'-'*52}")

    wg_vals = [
        r["within_grp_pop"]
        for r in summary
        if r["within_grp_pop"] is not None and r["model_type"] != "const_base"
    ]
    if wg_vals:
        _eval_label_e = "test" if EVAL_ON_TEST else "val"
        # wg_stats keys are tags; each value has n_orgs, n_years -- derive eval n from summary
        _eval_n_e = next(
            (
                r.get("vl_n") or r.get("val_n")
                for r in summary
                if r.get("vl_n") or r.get("val_n")
            ),
            "?",
        )
        _n_groups_e = max(
            (s.get("n_years", 0) * s.get("n_orgs", 0) for s in wg_stats.values()),
            default=0,
        )
        print(f"\n{'='*70}")
        print("[WG-POP DEBUG] E_plot_staged")
        print(f"  predictions file  : {PREDICTIONS_PATH}")
        print(f"  eval set          : {_eval_label_e}  n~={_eval_n_e}")
        print("  group key         : reporting_orgs + start_year")
        print(
            f"  n_orgs / n_years (max across tags): orgs={max((s.get('n_orgs',0) for s in wg_stats.values()), default=0)}  years={max((s.get('n_years',0) for s in wg_stats.values()), default=0)}"
        )
        print(f"  {'tag':<52s}  {'model':>10s}  {'wg_pop':>7s}")
        print(f"  {'-'*73}")
        for r in summary:
            if r["model_type"] != "const_base" and r["within_grp_pop"] is not None:
                print(
                    f"  {r['tag']:<52s}  {r['model_type']:>10s}  {r['within_grp_pop']:7.3f}"
                )
        print(f"  {'-'*73}")
        print(
            f"  {'mean (non-const)':<52s}  {'':>10s}  {float(np.nanmean(wg_vals)):7.3f}"
        )
        print(f"{'='*70}")
        print(f"\n{'-'*62}")
        print(f"wg_pop (non-const-baseline only, n={len(wg_vals)} of {len(summary)})")
        print(
            f"{'metric':8s}  {'mean':>7s}  {'median':>7s}  {'peak':>7s}  {'min':>7s}  n"
        )
        print(f"{'-'*62}")
        print(
            f"{'wg_pop':8s}  {np.mean(wg_vals):.3f}    {np.median(wg_vals):.3f}    "
            f"{np.max(wg_vals):.3f}    {np.min(wg_vals):.3f}    {len(wg_vals)}"
        )
        print(f"{'-'*62}")

        # Fraction of within-group pairs with differing true outcomes (across real-model tags)
        _tot_diff = sum(
            (wg_stats[r["tag"]].get("n_differing_pairs") or 0)
            for r in summary
            if r["model_type"] != "const_base" and r["tag"] in wg_stats
        )
        _tot_all = sum(
            (wg_stats[r["tag"]].get("n_total_pairs") or 0)
            for r in summary
            if r["model_type"] != "const_base" and r["tag"] in wg_stats
        )
        if _tot_all > 0:
            _diff_frac = _tot_diff / _tot_all
            print(
                f"\nDifferingOutcomePairsFrac (real-model tags, within-group): "
                f"{_diff_frac:.1%}  ({_tot_diff}/{_tot_all} pairs)"
            )

    # ---- Weighted accuracy: model vs constant-baseline ----
    _wt_model = _wt_base = _total_w = 0.0
    for r in summary:
        if r.get("model_type") == "const_base":
            continue  # stored val_acc uses train base rate; skip to avoid spurious improvement
        _vn = r.get("val_n") or 0
        _vp = r.get("val_n_pos") or 0
        if _vn <= 0:
            continue
        _base_acc = max(_vp, _vn - _vp) / _vn
        _acc = r.get("rf_acc") if r.get("rf_acc") is not None else _base_acc
        _wt_model += _acc * _vn
        _wt_base += _base_acc * _vn
        _total_w += _vn
    if _total_w > 0:
        _mean_model = _wt_model / _total_w
        _mean_base = _wt_base / _total_w
        n_real = sum(
            1
            for r in summary
            if r.get("model_type") != "const_base" and (r.get("val_n") or 0) > 0
        )
        print(f"\n{'-'*62}")
        print(
            f"WEIGHTED ACCURACY -- ALL {n_real} real-model tags (excl. const_base fallbacks)"
        )
        print("  Includes curated and non-curated tags; weights by val_n")
        print(f"  Constant baseline (majority class): {_mean_base:.1%}")
        print(f"  Chosen model:                       {_mean_model:.1%}")
        print(f"  Improvement:                        {_mean_model - _mean_base:+.1%}")
        print(f"{'-'*62}")

    # -- Curated-14 weighted accuracy ------------------------------------------
    curated_tags = [tag for _grp in TAG_GROUPS.values() for _lbl, tag in _grp]
    curated_rows = [r for r in summary if r["tag"] in curated_tags]
    _wt_m = _wt_b = _wt = 0.0
    for r in curated_rows:
        _vn = r.get("val_n") or 0
        if _vn <= 0:
            continue
        _vp = r.get("val_n_pos") or 0
        _base_acc = max(_vp, _vn - _vp) / _vn
        _acc = r.get("rf_acc") if r.get("rf_acc") is not None else _base_acc
        _wt_m += _acc * _vn
        _wt_b += _base_acc * _vn
        _wt += _vn
    if _wt > 0:
        n_cb = sum(1 for r in curated_rows if r["model_type"] == "const_base")
        cb_note = f"  ({n_cb} reverted to const_base)" if n_cb else ""
        n_curated_real = sum(1 for r in curated_rows if r["model_type"] != "const_base")
        print(f"\n{'-'*62}")
        print(
            f"WEIGHTED ACCURACY -- curated {len(curated_rows)} tags ({eval_label}){cb_note}"
        )
        print(
            f"  Tags reported in plots; includes const_base fallbacks ({n_curated_real} real models)"
        )
        print(f"  Constant baseline (majority class): {_wt_b/_wt:.1%}")
        print(f"  Chosen model:                       {_wt_m/_wt:.1%}")
        print(f"  Improvement:                        {(_wt_m-_wt_b)/_wt:+.1%}")
        print(f"{'-'*62}")

    # -- Curated vs non-curated: real-model tags only (no const_base) -------------
    def _tag_metrics(rows):
        """For a list of summary rows (non-const_base only), return lists of
        acc_improvement, val_pop, and brier_skill."""
        acc_imps, pops, briers = [], [], []
        for r in rows:
            if r["model_type"] == "const_base":
                continue
            _vn = r.get("val_n") or 0
            if _vn <= 0:
                continue
            _vp = r.get("val_n_pos") or 0
            _base_acc = max(_vp, _vn - _vp) / _vn
            _acc = r.get("rf_acc")
            if _acc is not None:
                acc_imps.append(_acc - _base_acc)
            if r.get("rf_pop") is not None:
                pops.append(r["rf_pop"])
            if r.get("rf_brier") is not None:
                briers.append(r["rf_brier"])
        return acc_imps, pops, briers

    curated_real = [r for r in curated_rows if r["model_type"] != "const_base"]
    noncurated_real = [
        r
        for r in summary
        if r["tag"] not in curated_tags and r["model_type"] != "const_base"
    ]
    c_acc, c_pop, c_brier = _tag_metrics(curated_real)
    n_acc, n_pop, n_brier = _tag_metrics(noncurated_real)

    def _fmt(vals):
        if not vals:
            return "N/A"
        return (
            f"mean={np.mean(vals):+.3f}  median={np.median(vals):+.3f}  n={len(vals)}"
        )

    print(f"\n{'-'*62}")
    print(
        f"CURATED ({len(curated_real)}) vs NON-CURATED ({len(noncurated_real)}) -- real models only"
    )
    print(f"  {'':25s}  {'curated':>30s}  {'non-curated':>30s}")
    print(f"  {'acc improvement':25s}  {_fmt(c_acc):>30s}  {_fmt(n_acc):>30s}")
    print(f"  {'val POP':25s}  {_fmt(c_pop):>30s}  {_fmt(n_pop):>30s}")
    print(f"  {'brier skill':25s}  {_fmt(c_brier):>30s}  {_fmt(n_brier):>30s}")
    print(f"{'-'*62}")

    # -- Brier score: curated 14 tags, all vs best 13 -------------------------
    brier_rows = []
    for r in curated_rows:
        _vn = r.get("val_n") or 0
        if _vn <= 0:
            continue
        baseline_brier = brier_for_model_type(raw_brier_by_tag, r["tag"], "const_base")
        if baseline_brier is None:
            # Fallback: use val_brier_base from results (train_rate constant predictor)
            baseline_brier = r.get("val_brier_base")
        model_brier = brier_for_model_type(
            raw_brier_by_tag, r["tag"], r.get("model_type", "rf+ET")
        )
        if model_brier is None:
            model_brier = baseline_brier
        brier_rows.append(
            {"tag": r["tag"], "model": model_brier, "baseline": baseline_brier}
        )

    if brier_rows:
        mean_model_14 = np.mean([b["model"] for b in brier_rows])
        mean_base_14 = np.mean([b["baseline"] for b in brier_rows])
        # best 13: drop the tag with the highest (worst) model brier
        brier_rows_13 = sorted(brier_rows, key=lambda b: b["model"])[:-1]
        worst_tag = max(brier_rows, key=lambda b: b["model"])["tag"]
        mean_model_13 = np.mean([b["model"] for b in brier_rows_13])
        mean_base_13 = np.mean([b["baseline"] for b in brier_rows_13])
        print(f"\n{'-'*62}")
        print("MEAN BRIER SCORE -- curated tags (lower = better)")
        print(
            f"  All {len(brier_rows)} tags:   baseline={mean_base_14:.4f}  model={mean_model_14:.4f}  diff={mean_model_14-mean_base_14:+.4f}"
        )
        print(
            f"  Best 13 tags: baseline={mean_base_13:.4f}  model={mean_model_13:.4f}  diff={mean_model_13-mean_base_13:+.4f}"
        )
        print(f"  (dropped worst: {worst_tag})")
        print(f"{'-'*62}")

    if skip_ci:
        return


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-ci",
        action="store_true",
        default=True,
        help="Skip the slow bootstrap CI plot (default: True)",
    )
    parser.add_argument(
        "--ci",
        dest="skip_ci",
        action="store_false",
        help="Run the slow bootstrap CI plot",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix inserted before the file extension for all input/output files. "
        "E.g. --suffix _manual loads tag_model_results_manual.json and saves "
        "tag_model_plots_manual.png. Default: '' (standard G_outcome_tag_train outputs).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on test set. Implies --suffix _test and uses test_idx instead of val_idx.",
    )
    parser.add_argument(
        "--nolimits",
        action="store_true",
        help="Plot nolimits run outputs. Implies --suffix _nolimits.",
    )
    args = parser.parse_args()

    global RESULTS_PATH, PREDICTIONS_PATH, OUT_PNG, OUT_CI_PNG, OUT_WG_PNG, EVAL_ON_TEST
    suffix_parts = []
    if args.nolimits:
        suffix_parts.append("nolimits")
        print("[nolimits] Using nolimits run outputs.")
    if args.test:
        suffix_parts.append("test")
        EVAL_ON_TEST = True
        print("WARNING: TEST SET Run! Evaluating on test set.")
    if suffix_parts:
        args.suffix = "_" + "_".join(suffix_parts)
    if args.suffix:
        RESULTS_PATH = OUT_DIR / f"tag_model_results{args.suffix}.json"
        PREDICTIONS_PATH = OUT_DIR / f"tag_predictions{args.suffix}.csv"
        OUT_PNG = OUT_DIR / f"tag_model_plots{args.suffix}.png"
        OUT_CI_PNG = OUT_DIR / f"tag_model_ci_plots{args.suffix}.png"
        OUT_WG_PNG = OUT_DIR / f"tag_wg_pop_vs_train_size{args.suffix}.png"

    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run D_train_tag_predictors.py first.")
        sys.exit(1)

    results = load_results(RESULTS_PATH)

    baseline_results = None
    if args.suffix:
        baseline_path = OUT_DIR / "tag_model_results.json"
        if baseline_path.exists():
            baseline_results = load_results(baseline_path)
        else:
            print(
                f"WARNING: baseline {baseline_path} not found -- Deltapop column will be omitted."
            )

    plot_from_results(results, skip_ci=args.skip_ci, baseline_results=baseline_results)


if __name__ == "__main__":
    main()
