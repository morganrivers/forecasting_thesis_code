"""
Unified script to compare multiple LLM forecast outputs by:
1. Grading them against actual outcomes (Validation Set A & 2)
2. Plotting grade histograms
3. Computing statistical metrics (RMSE, R^2, MAE, side_accuracy @ 3.5)
4. Paired bootstrap confidence intervals for metric differences (Validation Set A & 2)
5. Analyzing variance across multiple runs of same config (Set 3)

USAGE:
------
    python3 src/pipeline/F_llm_score_forecast_narratives.py

CONFIGURATION:
--------------
    Edit FORECAST_CONFIGS near the top to enable/disable which forecasts to process.
    Set "enabled": False to skip a forecast, True to process it.

PAIRED BOOTSTRAP COMPARISONS (Validation Set A & 2):
-----------------------------------------
    For selected method pairs, computes paired bootstrap 95% CIs for:
    - MAE difference (A - B): negative means A is better
    - RMSE difference (A - B): negative means A is better
    - SideAcc@3.5 difference (A - B): positive means A is better
    - Win rate: fraction of activities where A has lower absolute error

    Uses 10,000 bootstrap resamples with paired resampling of activity IDs.

SET 3 VARIANCE ANALYSIS:
-----------------------
    Set 3 analyzes variance between multiple runs of the same model.
    It does NOT perform grading - only analyzes consistency of ratings and metrics.
    Reports mean, std dev, min, max, and coefficient of variation for all metrics.

REQUIREMENTS:
    - Input files from data/rag_prompts_and_responses/
    - Ground truth ratings from data/merged_overall_ratings.jsonl
    - Outcomes from data/outputs_summary_expost.jsonl

OUTPUT:
    - Graded JSONL files: data/forecast_grades/grades_*.jsonl
    - Histograms: data/forecast_grades/plots/
    - Metrics printed to console
    - Paired bootstrap CIs printed to console
"""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from feature_engineering import (
    parse_last_line_label_after_forecast,
)
from leakage_risk import EXCLUDE_TEST_LEAKAGE_RISK, LEAKAGE_IDS_BY_SOURCE
from llm_extraction_and_grading import (
    loop_over_rows_to_call_model,
)
from llm_grading_utils import (
    calculate_metrics,
    extract_forecast_ratings,
    load_ground_truth_ratings,
    load_jsonl_by_activity_id,
)
from ml_models import bootstrap_ci
from scoring_metrics import (
    pairwise_ordering_prob_excl_ties,
    side_accuracy,
    within_group_pairwise_ordering_prob,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

FORECAST_CONFIGS = {
    "set1_300_samples": [
        {
            "name": "gemini-3-pro (KNN+RAG+S1+S2)",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_gemini3pro_val_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "deepseek-R1 (KNN+RAG+S1+S2)",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_deepseek_minimal_val_s3_call_1.jsonl",
            "enabled": True,
            "leakage_source_key": "deepseek_minimal_val_forecast",
        },
        {
            "name": "deepseek-R1 RF forced (KNN+RAG+S1+S2)",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_forced_rf_deepseek_val_forced_rf_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "deepseek-R1 RF forced (no KNN, no RAG, no S1, no S2)",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_no_knn_no_rag_forced_rf_deepseek_val_no_knn_no_rag_forced_rf_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "deepseek-R1, (KNN and RAG added, no S1, no S2)",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_deepseek_minimal_val_no_goodbadcalls_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "deepseek-R1 (no KNN, no RAG, no S1, no S2)",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_no_knn_no_rag_deepseek_minimal_val_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "deepseek-R1 (minimal prompt)",
            "file": "outputs_onlysummary_no_knn_no_rag_onlysummary_no_knn_no_rag_s3_call_1.jsonl",
            "enabled": True,
        },
    ],
    "set2_171_samples": [
        {
            "name": "gemini-2.5-flash KNN+RAG+S1+S2 finetuned",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_vertex_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "gemini-2.5-flash KNN+RAG+S1+S2",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_no_vertex_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "gemini-3-pro s3, KNN+RAG+S1+S2",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_GEMINI3PRO_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "deepseek-R1 KNN+RAG+S1+S2",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_deepseek_with_stages_s3_call_1.jsonl",
            "enabled": True,
        },
        {
            "name": "DeepSeek s3 KNN+RAG, no s1 or s2",
            "file": "outputs_exactly_like_halawi_et_al_rag_added_deepseek_s3_call_1.jsonl",
            "enabled": True,
        },
    ],
    "set3_variance_analysis": [],
}

OUTCOMES_JSONL = Path("../../data/outputs_summary_expost.jsonl")
RATINGS_JSONL = Path("../../data/merged_overall_ratings.jsonl")
DATA_DIR = Path("../../data/rag_prompts_and_responses")
GRADES_OUTPUT_DIR = Path("../../data/forecast_grades")


# ---------------------------------------------------------------------------
# PREPROCESSING: EXTRACT RATINGS AND REMOVE LAST LINES
# ---------------------------------------------------------------------------


def extract_ratings_and_remove_last_lines(
    forecasts: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """
    For each forecast:
    1. Extract numeric rating from last line using parse_last_line_label_after_forecast
    2. Remove the last line

    Returns (forecasts_without_last_line, ratings_dict)
    """
    forecasts_cleaned = {}
    ratings = {}

    for aid, text in forecasts.items():
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue

        # Extract rating from full text (parser finds last line internally)
        rating = parse_last_line_label_after_forecast(text, record={"activity_id": aid})
        if rating is not None:
            ratings[aid] = rating

        # Remove last line and rejoin
        forecast_without_last = "\n".join(lines[:-1])
        forecasts_cleaned[aid] = forecast_without_last

    return forecasts_cleaned, ratings


# ---------------------------------------------------------------------------
# OUTCOME LOADING
# ---------------------------------------------------------------------------


def load_outcomes(outcomes_path: Path) -> dict[str, str]:
    """Load outcomes JSONL file."""
    return load_jsonl_by_activity_id(outcomes_path)


def load_risks_as_forecasts(risks_path: Path, set_ids: set[str]) -> dict[str, str]:
    """Load risks JSONL, return {activity_id: response_text} filtered to set_ids."""
    risks = {}
    with open(risks_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                aid = data.get("activity_id")
                if aid and aid in set_ids:
                    risks[aid] = data.get("response_text", "")
            except json.JSONDecodeError:
                continue
    return risks


# ---------------------------------------------------------------------------
# GRADING
# ---------------------------------------------------------------------------


def build_grading_prompts(
    forecasts: dict[str, str],
    outcomes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """
    Build prompts that ask gemini-2.5-flash-lite to grade each forecast.
    """
    prompts: dict[str, dict[str, Any]] = {}

    # Only grade activities where we have both forecast and outcome
    common_aids = set(forecasts.keys()) & set(outcomes.keys())

    print(f"Found {len(forecasts)} forecasts, {len(outcomes)} outcomes")
    print(f"Will grade {len(common_aids)} activities with both\n")

    for aid in sorted(common_aids):
        forecast_text = forecasts[aid]
        outcome_text = outcomes[aid]

        prompt_text = f"""You are grading the quality of an ex-ante forecast against what actually happened (ex-post outcomes), as relates to its overall success.

Grade the accuracy of the forecast on a scale from A+ to F.
Were the considerations in the forecast validated by the true outcome?
Were there key outcomes related to the core goals of the activity that were forecasted? (very good)
Were there key forecasted outcomes that didn't turn out like that? (bad)
Was the attention of the forecast focused on the key drivers, or missing the mark?

Grading scale:
- A+/A/A-: Excellent forecast, highly accurate, attention on key drivers, multiple major events forecasted accurately.
- B+/B/B-: Good forecast, mostly accurate. A mix of correct and incorrect, but at least one major outcome was forecasted. At least one key driver identified.
- C+/C/C-: Adequate forecast, partially accurate or reasonable. Focus was adequate. Major outcomes incorrect, but some smaller aspects were correct.
- D+/D/D-: Poor forecast, although perhaps one or two small correct things. Mostly inaccurate. Wrong focus on drivers.
- F: Failed forecast, completely wrong or unsupported.

FORECAST (what was predicted before the activity):
{forecast_text}

ACTUAL OUTCOMES (what happened after completion):
{outcome_text}

Provide:
1. Analysis (3-5 sentences) comparing forecast to outcomes
2. Final grade on the last line in format: GRADE: [letter grade]

Respond only in English."""

        prompts[aid] = {
            "system_msg": (
                "You are an experienced international aid evaluator. "
                "Grade the following forecast for how well it adhered to the true outcomes."
            ),
            "prompt": prompt_text,
            "prompt_type": "forecast_grading",
        }

    if prompts:
        print("Sample grading prompt (first activity):")
        sample = list(prompts.values())[0]
        print(sample["prompt"][:500] + "...\n")

    return prompts


def grade_forecast_set(
    forecast_path: Path,
    outcomes_path: Path,
    output_path: Path,
    execpool: ThreadPoolExecutor,
) -> None:
    """
    Grade forecasts against outcomes.
    1. Extract ratings from last lines of forecasts
    2. Remove last lines from forecasts
    3. Grade the cleaned forecasts against outcomes using gemini-2.5-flash-lite

    Note: loop_over_rows_to_call_model automatically skips already-graded IDs.
    """
    print("\nLoading forecasts and outcomes...")
    raw_forecasts = load_jsonl_by_activity_id(forecast_path)
    outcomes = load_outcomes(outcomes_path)

    print("Extracting ratings and removing last lines...")
    forecasts_cleaned, extracted_ratings = extract_ratings_and_remove_last_lines(
        raw_forecasts
    )

    print(f"  Extracted {len(extracted_ratings)} ratings")
    print(f"  Cleaned {len(forecasts_cleaned)} forecasts")

    print("Building grading prompts...")
    prompts = build_grading_prompts(forecasts_cleaned, outcomes)

    if not prompts:
        print("ERROR: No common activities found between forecasts and outcomes!")
        return

    # Build rows for loop_over_rows_to_call_model
    rows = [{"activity_id": aid} for aid in sorted(prompts.keys())]

    # Count existing grades by activity ID
    already_graded_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        aid = data.get("activity_id")
                        if aid:
                            already_graded_ids.add(aid)
                    except json.JSONDecodeError:
                        continue

    # Determine which IDs need grading
    all_ids_to_grade = set(prompts.keys())
    missing_ids = all_ids_to_grade - already_graded_ids

    print(f"\n{'=' * 80}")
    print("GRADING SUMMARY:")
    print(f"  Total activities with forecasts+outcomes: {len(all_ids_to_grade)}")
    print(f"  Already graded:                           {len(already_graded_ids)}")
    print(f"  Missing grades (will run):                {len(missing_ids)}")
    print(f"{'=' * 80}")

    if len(missing_ids) == 0:
        print(f"All activities already graded: {output_path}")
        return

    input(f"\nPress Enter to proceed with grading {len(missing_ids)} activities...")

    print(f"\nCalling gemini-2.5-flash-lite for {len(missing_ids)} missing grades...")
    print("(loop_over_rows will skip already-graded IDs automatically)")

    asyncio.run(
        loop_over_rows_to_call_model(
            str(output_path),
            rows,
            prompts,
            response_schema=None,
            execpool=execpool,
            model="gemini-2.5-flash-lite",
        )
    )

    print(f"Grades written to: {output_path}")


def grade_risks(
    risks: dict[str, str],
    outcomes_path: Path,
    output_path: Path,
    execpool: ThreadPoolExecutor,
) -> None:
    """
    Grade risk summaries against outcomes using the same grading pipeline.
    No rating extraction or last-line removal -- risks are pure text.
    """
    outcomes = load_outcomes(outcomes_path)
    print("Building grading prompts for risks...")
    prompts = build_grading_prompts(risks, outcomes)

    if not prompts:
        print("ERROR: No common activities found between risks and outcomes!")
        return

    rows = [{"activity_id": aid} for aid in sorted(prompts.keys())]

    already_graded_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        aid = data.get("activity_id")
                        if aid:
                            already_graded_ids.add(aid)
                    except json.JSONDecodeError:
                        continue

    missing_ids = set(prompts.keys()) - already_graded_ids
    print(f"  Total risks with outcomes: {len(prompts)}")
    print(f"  Already graded:            {len(already_graded_ids)}")
    print(f"  Missing (will run):        {len(missing_ids)}")

    if len(missing_ids) == 0:
        print(f"All risks already graded: {output_path}")
        return

    input(f"\nPress Enter to proceed with grading {len(missing_ids)} risk summaries...")

    print(f"\nCalling gemini-2.5-flash-lite for {len(missing_ids)} missing grades...")
    asyncio.run(
        loop_over_rows_to_call_model(
            str(output_path),
            rows,
            prompts,
            response_schema=None,
            execpool=execpool,
            model="gemini-2.5-flash-lite",
        )
    )

    print(f"Risk grades written to: {output_path}")


# ---------------------------------------------------------------------------
# ANALYSIS AND COMPARISON
# ---------------------------------------------------------------------------


def grade_to_numeric(grade: str) -> float:
    """Convert letter grade to percentage scale (55-97)."""
    grade_map = {
        "A+": 97,
        "A": 95,
        "A-": 92,
        "B+": 88,
        "B": 85,
        "B-": 82,
        "C+": 78,
        "C": 75,
        "C-": 72,
        "D+": 68,
        "D": 65,
        "D-": 62,
        "F": 55,
    }
    return grade_map.get(grade.strip(), None)


def _metric_sideacc_3_5(y_true, y_pred, thr=3.5):
    return side_accuracy(y_true, y_pred, threshold=thr)


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# pairwise_ordering_prob removed -- use pairwise_ordering_prob_excl_ties from scoring_metrics


def paired_bootstrap_metric_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Paired bootstrap CI for (metric(A) - metric(B)) using resampling of items.
    Returns mean diff and percentile CI.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = np.arange(n)

    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        diffs[b] = metric_fn(y_true[s], y_pred_a[s]) - metric_fn(y_true[s], y_pred_b[s])

    diffs.sort()
    return {
        "diff_mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def paired_win_rate(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> float:
    """
    Fraction of items where A has lower absolute error than B.
    Ties count as 0.5.
    """
    ea = np.abs(y_true - y_pred_a)
    eb = np.abs(y_true - y_pred_b)
    return float(np.mean((ea < eb) + 0.5 * (ea == eb)))


def load_grades_with_ids(path: Path) -> dict[str, float]:
    """Load grades JSONL and return dict of activity_id -> numeric grade."""

    grades = {}
    if not path.exists():
        return grades

    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                aid = data.get("activity_id")
                response = data.get("response_text", "")

                # Extract grade from response (format: "GRADE: A+" or "GRADE: ** A **")
                if "GRADE:" in response:
                    grade_text = response.split("GRADE:")[-1].strip()

                    # Remove markdown formatting (**, *, etc.)
                    grade_text = grade_text.replace("**", "").replace("*", "").strip()

                    # Extract first word (the actual grade)
                    grade_part = grade_text.split()[0] if grade_text.split() else ""

                    numeric = grade_to_numeric(grade_part)
                    if numeric is not None:
                        grades[aid] = numeric
            except (json.JSONDecodeError, ValueError, IndexError):
                continue

    return grades


def plot_2pane_comparison(
    results_a: dict[str, dict[str, Any]],
    set_name_a: str,
    output_path: Path,
    human_proxy_mean_grade: float = None,
) -> None:
    """
    2-pane plot:
      Top:    Mean Grade vs R^2
      Bottom: Mean Grade vs Pairwise Ord Prob
    """

    def count_pred_ties(y_pred_arr):
        y_pred_arr = np.asarray(y_pred_arr, dtype=float)
        if len(y_pred_arr) < 2:
            return 0.0
        i_idx, j_idx = np.triu_indices(len(y_pred_arr), k=1)
        n_ties = int(np.sum(y_pred_arr[i_idx] == y_pred_arr[j_idx]))
        n_pairs = len(i_idx)
        return 100.0 * n_ties / n_pairs if n_pairs > 0 else 0.0

    def extract(results, metric_key):
        names, x, y = [], [], []
        for name, data in results.items():
            if metric_key == "pairwise_within_group":
                val = data.get("pairwise_within_group", float("nan"))
                if np.isnan(val):
                    continue
            else:
                val = data["metrics"][metric_key]
            names.append(name)
            x.append(data["mean_grade"])
            y.append(val)
        return names, np.array(x), np.array(y)

    panel_data = {
        "a_r2": extract(results_a, "r2"),
        "a_pw": extract(results_a, "pairwise_within_group"),
    }

    # Override forced points using OOS val RF predictions (train-only model).
    # Prefer rf_oos_val_predictions.csv (train-only, true OOS) over pred_rf in
    # best_model_predictions.csv (which was trained with val in train -- inflated).
    _oos_path = Path("../../data/rf_oos_val_predictions.csv")
    if not _oos_path.exists():
        raise FileNotFoundError(
            f"{_oos_path} not found. "
            "Run src/D_data_analysis/D_overall_rating_generate_oos_predictions.py first to generate OOS predictions."
        )
    _rf_df = pd.read_csv(_oos_path)
    pred_rf_map = dict(zip(_rf_df["activity_id"], _rf_df["pred_rf_oos"], strict=False))
    print(
        f"  [RF forced] using OOS val predictions from {_oos_path.name} ({len(pred_rf_map)} activities)"
    )
    panel_override_info = {
        "a_r2": (results_a, True),
        "a_pw": (results_a, False),
    }
    for key, (names, _x, y) in panel_data.items():
        results_dict, is_r2 = panel_override_info[key]
        for i, name in enumerate(names):
            if "forced" in name.lower() and name in results_dict:
                ids = results_dict[name]["graded_ids_sorted"]
                y_true_arr = results_dict[name]["y_true"]
                groups_arr = results_dict[name]["groups"]
                y_rf = np.array([pred_rf_map.get(aid, float("nan")) for aid in ids])
                valid = ~np.isnan(y_rf)
                if valid.sum() >= 2:
                    if is_r2:
                        y[i] = _r2(y_true_arr[valid], y_rf[valid])
                    else:
                        y[i] = within_group_pairwise_ordering_prob(
                            y_true_arr[valid], y_rf[valid], groups_arr[valid]
                        )["prob"]

    # Compute avg pred tie% and truth tie% across all methods (for y-axis label)
    pred_tie_pcts = []
    truth_tie_pct = None
    for _name, data in results_a.items():
        y_pred_arr = data.get("y_pred", np.array([]))
        y_true_arr = data.get("y_true", np.array([]))
        if len(y_pred_arr) >= 2:
            n_pairs = len(y_pred_arr) * (len(y_pred_arr) - 1) // 2
            i_idx, j_idx = np.triu_indices(len(y_pred_arr), k=1)
            pred_tie_pcts.append(
                100.0 * np.sum(y_pred_arr[i_idx] == y_pred_arr[j_idx]) / n_pairs
            )
            if truth_tie_pct is None and len(y_true_arr) >= 2:
                truth_tie_pct = (
                    100.0 * np.sum(y_true_arr[i_idx] == y_true_arr[j_idx]) / n_pairs
                )
    avg_pred_tie = float(np.mean(pred_tie_pcts)) if pred_tie_pcts else float("nan")

    r2_vals = panel_data["a_r2"][2]
    pw_vals = panel_data["a_pw"][2]
    r2_pad = (r2_vals.max() - r2_vals.min()) * 0.25 if len(r2_vals) else 0.1
    pw_pad = (pw_vals.max() - pw_vals.min()) * 0.25 if len(pw_vals) else 0.1
    r2_ylim = (
        (r2_vals.min() - r2_pad, r2_vals.max() + r2_pad) if len(r2_vals) else (0, 1)
    )
    pw_ylim = (
        (pw_vals.min() - pw_pad, pw_vals.max() + pw_pad) if len(pw_vals) else (0.5, 1)
    )

    pw_ylabel = "Pairwise Within Group"
    if not np.isnan(avg_pred_tie) and truth_tie_pct is not None:
        pw_ylabel += (
            f"\n(avg pred ties: {avg_pred_tie:.1f}%, truth ties: {truth_tie_pct:.1f}%)"
        )

    fig, axes = plt.subplots(2, 1, figsize=(14, 18))

    panel_specs = [
        (axes[0], panel_data["a_r2"], "R^2 (Forecast vs Ground Truth)", r2_ylim, True),
        (axes[1], panel_data["a_pw"], pw_ylabel, pw_ylim, False),
    ]

    for ax, (names, x, y), ylabel, ylim, is_top in panel_specs:
        if len(names) < 2:
            ax.set_title("(need >= 2 methods)", fontsize=28, fontweight="bold")
            continue

        for i, name in enumerate(names):
            name_lower = name.lower()
            marker = "^" if "gemini-3-pro" in name_lower else "o"
            color = "green" if "forced" in name_lower else "steelblue"
            ax.scatter(
                x[i],
                y[i],
                alpha=0.7,
                s=180,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidth=1.5,
                zorder=3,
            )
        texts = []
        for i, name in enumerate(names):
            name_lower = name.lower()
            if "forced" in name_lower:
                # Manual placement for green points -- offset relative to ylim so both panels look identical
                yrange = ylim[1] - ylim[0]
                y_offset = yrange * 0.04 if "no knn" in name_lower else yrange * 0.08
                if not is_top and "no knn" in name_lower:
                    text_y = 0.61
                else:
                    text_y = y[i] + y_offset
                ax.annotate(
                    name,
                    xy=(x[i], y[i]),
                    xytext=(x[i] - 2.5, text_y),
                    fontsize=20,
                    fontweight="bold",
                    alpha=0.9,
                    arrowprops=dict(arrowstyle="->", color="red"),
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )
            elif "minimal prompt" in name_lower and not is_top:
                # Manual placement in bottom pane: above the point, inside xlim
                yrange = ylim[1] - ylim[0]
                ax.annotate(
                    name,
                    xy=(x[i], y[i]),
                    xytext=(x[i], ylim[1] - yrange * 0.08),
                    fontsize=20,
                    fontweight="bold",
                    alpha=0.9,
                    arrowprops=dict(arrowstyle="->", color="red"),
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )
            elif "gemini" in name_lower and not is_top:
                ax.text(
                    x[i] - 0.15,
                    y[i],
                    name,
                    fontsize=20,
                    fontweight="bold",
                    alpha=0.9,
                    ha="right",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )
            elif "knn and rag added" in name_lower and not is_top:
                # Manual placement in bottom pane: below its point to avoid RF forced labels above
                yrange = ylim[1] - ylim[0]
                ax.annotate(
                    name,
                    xy=(x[i], y[i]),
                    xytext=(x[i] - 2.0, y[i] - yrange * 0.12),
                    fontsize=20,
                    fontweight="bold",
                    alpha=0.9,
                    arrowprops=dict(arrowstyle="->", color="red"),
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )
            else:
                t = ax.text(
                    x[i],
                    y[i],
                    name,
                    fontsize=20,
                    fontweight="bold",
                    alpha=0.9,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.7,
                    ),
                )
                texts.append(t)
        if texts:
            adjust_text(
                texts,
                x,
                y,
                ax=ax,
                expand=(1.5, 3.0),
                force_text=(0.8, 3.0),
                lw=0.5,
                arrowprops=dict(arrowstyle="->", color="red"),
            )
        ax.set_ylim(ylim)
        ax.set_xlim(left=84)
        if is_top:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Mean Forecast Grade (55-97 scale)", fontsize=26)
        ax.set_ylabel(ylabel, fontsize=26)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=24)

    fig.suptitle(
        "Forecast Method Comparison: Mean Grade vs Accuracy Metrics",
        fontsize=36,
        fontweight="bold",
        y=1.01,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="steelblue",
            markeredgecolor="black",
            markersize=14,
            label="deepseek",
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="steelblue",
            markeredgecolor="black",
            markersize=14,
            label="gemini-3-pro",
            linestyle="None",
        ),
        Patch(facecolor="steelblue", edgecolor="black", label="LLM only"),
        Patch(facecolor="green", edgecolor="black", label="RF forced"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        fontsize=22,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.tight_layout(rect=[0.03, 0.05, 1, 0.98], h_pad=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"  2-pane comparison plot saved: {output_path}")


def generate_latex_table(
    results: dict[str, dict[str, Any]],
    set_name: str,
    enabled_configs: list[dict[str, Any]],
) -> str:
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{" + f"Forecast Performance Metrics - {set_name}" + "}")
    latex.append("\\begin{tabular}{l r r r r r r r r}")
    latex.append("\\hline")
    latex.append(
        "Method & N & $R^2$ & PairOrd & RMSE & MAE & SideAcc@3.5 & Mean Grade & Median Grade \\\\"
    )
    latex.append("\\hline")

    for config in enabled_configs:
        name = config["name"]
        if name not in results:
            continue

        metrics = results[name]["metrics"]
        mean_grade = results[name]["mean_grade"]
        median_grade = results[name]["median_grade"]

        # Escape special characters in method name
        escaped_name = name.replace("&", "\\&").replace("_", "\\_")

        pairwise_val = results[name].get("pairwise", float("nan"))
        latex.append(
            f"{escaped_name} & {metrics['n']} & "
            f"{metrics['r2']:.4f} & {pairwise_val:.4f} & {metrics['rmse']:.4f} & "
            f"{metrics['mae']:.4f} & {metrics['side_accuracy_3_5']:.4f} & "
            f"{mean_grade:.2f} & {median_grade:.2f} \\\\"
        )

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def ensure_graded_output(
    forecast_path: Path,
    outcomes_path: Path,
    grades_output_path: Path,
    execpool: ThreadPoolExecutor,
) -> Path:
    """
    Ensure graded output exists. Grade missing IDs if needed.
    Always checks ID-by-ID, not just file existence.
    loop_over_rows_to_call_model automatically skips already-graded IDs.

    Returns path to graded output.
    """
    # Always call grade_forecast_set - it will check which IDs are missing
    # and only grade those (loop_over_rows_to_call_model skips existing IDs)
    grade_forecast_set(
        forecast_path,
        outcomes_path,
        grades_output_path,
        execpool,
    )

    return grades_output_path


def compare_forecast_set(
    set_name: str,
    forecast_configs: list[dict[str, Any]],
    ground_truth_ratings: pd.Series,
    outcomes_path: Path,
    data_dir: Path,
    grades_output_dir: Path,
    execpool: ThreadPoolExecutor,
    common_activity_ids: set = None,
    set_defining_file: str = None,
    org_labels: pd.Series = None,
    restrict_ids: set = None,
) -> None:
    """Compare forecasts in a set across multiple dimensions."""
    print("\n" + "=" * 80)
    print(f"COMPARISON SET: {set_name}")
    print("=" * 80)

    enabled_configs = [c for c in forecast_configs if c["enabled"]]
    print(f"Enabled forecasts: {len(enabled_configs)}")

    set_ids = None
    if set_defining_file:
        set_defining_path = data_dir / set_defining_file
        if set_defining_path.exists():
            set_ids = set(load_jsonl_by_activity_id(set_defining_path).keys())
            print(f"Set-defining file: {set_defining_file}")
            print(f"  Activities in set: {len(set_ids)}")
        else:
            print(f"WARNING: Set-defining file not found: {set_defining_file}")

    outcomes = load_outcomes(outcomes_path)
    print("\nDEBUG - Data availability:")
    print(f"  Ground truth ratings: {len(ground_truth_ratings)}")
    print(f"  Outcomes available:   {len(outcomes)}")
    if set_ids:
        print(f"  Set IDs:              {len(set_ids)}")
        print(
            f"  Set & Ground truth:   {len(set_ids & set(ground_truth_ratings.index))}"
        )
        print(f"  Set & Outcomes:       {len(set_ids & set(outcomes.keys()))}")
        print(
            f"  Set & GT & Outcomes:  {len(set_ids & set(ground_truth_ratings.index) & set(outcomes.keys()))}"
        )

    results = {}
    grade_lists = {}
    graded_paths = {}
    for config in enabled_configs:
        name = config["name"]
        filename = config["file"]
        forecast_path = data_dir / filename

        print(f"\n--- Grading: {name} ---")

        if not forecast_path.exists():
            print(f"ERROR: File not found: {forecast_path}")
            sys.exit(1)

        grades_output_path = grades_output_dir / f"grades_{filename}"
        ensure_graded_output(forecast_path, outcomes_path, grades_output_path, execpool)
        graded_paths[name] = (forecast_path, grades_output_path)

    per_method_data = {}
    for config in enabled_configs:
        name = config["name"]
        if name not in graded_paths:
            continue
        forecast_path, grades_output_path = graded_paths[name]

        print(f"\n--- Loading: {name} ---")
        ratings = extract_forecast_ratings(
            forecast_path, parser=parse_last_line_label_after_forecast
        )
        print(f"  Extracted {len(ratings)} ratings from file")

        if set_ids:
            ratings = ratings[ratings.index.isin(set_ids)]
            print(f"  Filtered to set IDs: {len(ratings)} ratings")
        if common_activity_ids and not set_ids:
            ratings = ratings[ratings.index.isin(common_activity_ids)]
            print(f"  Filtered to common activity IDs: {len(ratings)} ratings")

        grades_numeric = load_grades_with_ids(grades_output_path)
        print(f"  Extracted {len(grades_numeric)} grades")

        valid_ids = (
            set(ratings.index)
            & set(ground_truth_ratings.index)
            & set(grades_numeric.keys())
        )
        valid_ids_ranking = set(ratings.index) & set(ground_truth_ratings.index)
        if restrict_ids is not None:
            valid_ids = valid_ids & restrict_ids
            valid_ids_ranking = valid_ids_ranking & restrict_ids

        if EXCLUDE_TEST_LEAKAGE_RISK:
            _src_key = config.get("leakage_source_key")
            if _src_key and _src_key in LEAKAGE_IDS_BY_SOURCE:
                _leaky = LEAKAGE_IDS_BY_SOURCE[_src_key]
                _before = len(valid_ids)
                valid_ids -= _leaky
                valid_ids_ranking -= _leaky
                print(
                    f"  [leakage] dropped {_before - len(valid_ids)} activities "
                    f"with leakage from source '{_src_key}'"
                )

        print(f"  Valid IDs (ratings & GT & grades): {len(valid_ids)}")
        print(
            f"  Valid IDs for ranking (ratings & GT, no grade req): {len(valid_ids_ranking)}"
        )

        per_method_data[name] = {
            "ratings": ratings,
            "grades_numeric": grades_numeric,
            "valid_ids": valid_ids,
            "valid_ids_ranking": valid_ids_ranking,
            "grades_output_path": grades_output_path,
            "forecast_path": forecast_path,
        }

    # Compute strict overlap across ALL methods
    if per_method_data:
        strict_ids = sorted(
            set.intersection(*[d["valid_ids"] for d in per_method_data.values()])
        )
        strict_ids_ranking = sorted(
            set.intersection(
                *[d["valid_ids_ranking"] for d in per_method_data.values()]
            )
        )
    else:
        strict_ids = []
        strict_ids_ranking = []
    print(f"\nStrict overlap across all methods: {len(strict_ids)} activities")
    print(
        f"Strict overlap for ranking (no grade req): {len(strict_ids_ranking)} activities"
    )

    # Save strict_ids to the shared eval-set-sizes directory for cross-script comparison
    _eval_dir = Path("../../data/eval_set_sizes")
    _eval_dir.mkdir(parents=True, exist_ok=True)
    _ai_splits = pd.DataFrame(
        {"activity_id": list(strict_ids), "set_name": set_name, "split": "val"}
    )
    _ai_out = _eval_dir / f"ai_forecasting_splits_{set_name}.csv"
    _ai_splits.to_csv(_ai_out, index=False)
    print(f"[eval_set_sizes] ai_forecasting strict_ids ({set_name}) saved to {_ai_out}")

    # Save combined forecast+grade files for each method (forecast text + grading reasoning + extracted grade)
    _iati_grades_dir = Path.home() / "Code" / "iati_extractions" / "llm_grades"
    _iati_grades_dir.mkdir(parents=True, exist_ok=True)
    for _name, _d in per_method_data.items():
        _forecasts = load_jsonl_by_activity_id(_d["forecast_path"])
        _grades_by_id = {}
        if _d["grades_output_path"].exists():
            with open(_d["grades_output_path"]) as _gf:
                for _line in _gf:
                    if not _line.strip():
                        continue
                    try:
                        _obj = json.loads(_line)
                        _aid = _obj.get("activity_id")
                        if _aid:
                            _grades_by_id[_aid] = _obj
                    except json.JSONDecodeError:
                        continue
        _combined = []
        for _aid in sorted(set(_forecasts) | set(_grades_by_id)):
            _grade_obj = _grades_by_id.get(_aid, {})
            _grade_resp = _grade_obj.get("response_text", "")
            _grade_numeric = _d["grades_numeric"].get(_aid)
            _grade_letter = None
            import re as _re

            _gm = _re.findall(
                r"GRADE:\s*([A-F][+-]?)(?=\s|$)", _grade_resp, _re.IGNORECASE
            )
            if _gm:
                _grade_letter = _gm[-1].upper()
            _combined.append(
                {
                    "activity_id": _aid,
                    "method_name": _name,
                    "forecast_text": _forecasts.get(_aid, ""),
                    "grading_reasoning": _grade_resp,
                    "extracted_grade_letter": _grade_letter,
                    "extracted_grade_numeric": _grade_numeric,
                }
            )
        _safe_name = _re.sub(r"[^\w]", "_", _name)[:80]
        _out_path = _iati_grades_dir / f"combined_{_safe_name}.jsonl"
        with open(_out_path, "w") as _out:
            for _row in _combined:
                _out.write(json.dumps(_row) + "\n")
        print(
            f"  Saved combined forecast+grade for '{_name}': {len(_combined)} entries -> {_out_path.name}"
        )

    # Compute metrics using strict overlap
    for config in enabled_configs:
        name = config["name"]
        if name not in per_method_data:
            continue

        d = per_method_data[name]
        ratings = d["ratings"]
        grades_numeric = d["grades_numeric"]
        grades_output_path = d["grades_output_path"]
        forecast_path = d["forecast_path"]

        graded_ids_sorted = strict_ids
        graded_ids = set(strict_ids)

        print(f"\n--- Metrics: {name} ({len(graded_ids_sorted)} activities) ---")

        # Extract grades list for histograms - filtered to strict overlap only
        grades_for_graded = []
        with open(grades_output_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    aid = data.get("activity_id")
                    if aid in graded_ids:
                        response = data.get("response_text", "")
                        if "GRADE:" in response:
                            grade_text = response.split("GRADE:")[-1].strip()
                            grade_text = (
                                grade_text.replace("**", "").replace("*", "").strip()
                            )
                            grade_part = (
                                grade_text.split()[0] if grade_text.split() else ""
                            )
                            if grade_part:
                                grades_for_graded.append(grade_part)
                except (json.JSONDecodeError, IndexError):
                    continue

        grade_lists[name] = grades_for_graded

        if len(graded_ids_sorted) == 0:
            print(f"WARNING: No activities in strict overlap for {name}")
            continue

        # Pairwise/ranking metrics: use this method's own full coverage (no cross-method intersection)
        ranking_ids_sorted = sorted(d["valid_ids_ranking"])
        y_pred_rank = ratings[ranking_ids_sorted].astype(float).to_numpy()
        y_true_rank = ground_truth_ratings[ranking_ids_sorted].astype(float).to_numpy()
        if org_labels is not None:
            groups_rank = np.array(
                [org_labels.get(aid, "unknown") for aid in ranking_ids_sorted]
            )
        else:
            groups_rank = np.zeros(len(ranking_ids_sorted), dtype=int)

        # Grade-based metrics use the smaller graded set
        y_pred = ratings[graded_ids_sorted].astype(float).to_numpy()
        y_true = ground_truth_ratings[graded_ids_sorted].astype(float).to_numpy()

        # Build group labels aligned to graded_ids_sorted
        if org_labels is not None:
            groups = np.array(
                [org_labels.get(aid, "unknown") for aid in graded_ids_sorted]
            )
        else:
            groups = np.zeros(
                len(graded_ids_sorted), dtype=int
            )  # single group fallback

        metrics = calculate_metrics(y_true, y_pred)
        pairwise_val = pairwise_ordering_prob_excl_ties(y_true_rank, y_pred_rank)
        wg_pairwise_val = within_group_pairwise_ordering_prob(
            y_true_rank, y_pred_rank, groups_rank
        )["prob"]
        r2_ci = bootstrap_ci(y_true_rank, y_pred_rank, _r2, n_bootstrap=100)
        pairwise_ci = bootstrap_ci(
            y_true_rank, y_pred_rank, pairwise_ordering_prob_excl_ties, n_bootstrap=100
        )

        grades_for_graded_numeric = [
            grades_numeric[aid] for aid in graded_ids_sorted if aid in grades_numeric
        ]
        mean_grade = (
            np.mean(grades_for_graded_numeric) if grades_for_graded_numeric else 0.0
        )
        median_grade = (
            np.median(grades_for_graded_numeric) if grades_for_graded_numeric else 0.0
        )

        results[name] = {
            "config": config,
            "ratings": ratings,
            "grades_numeric": grades_numeric,
            "metrics": metrics,
            "mean_grade": mean_grade,
            "median_grade": median_grade,
            "graded_ids_sorted": graded_ids_sorted,
            "y_true": y_true,
            "y_pred": y_pred,
            "groups": groups,
            "pairwise": pairwise_val,
            "pairwise_within_group": wg_pairwise_val,
            "r2_ci": r2_ci,
            "pairwise_ci": pairwise_ci,
        }

        {
            aid: grades_numeric[aid]
            for aid in graded_ids_sorted
            if aid in grades_numeric
        }

        if metrics["n"] > 0:
            print(f"\n  Metrics (strict overlap n={metrics['n']}):")
            print(
                f"    R^2:                {metrics['r2']:.4f}  (95% CI [{r2_ci['lower']:.4f}, {r2_ci['upper']:.4f}])"
            )
            print(
                f"    Pairwise Ord Prob: {pairwise_val:.4f}  (95% CI [{pairwise_ci['lower']:.4f}, {pairwise_ci['upper']:.4f}])"
            )
            print(f"    RMSE:              {metrics['rmse']:.4f}")
            print(f"    MAE:               {metrics['mae']:.4f}")
            print(f"    Side Accuracy@3.5: {metrics['side_accuracy_3_5']:.4f}")
            print(f"    Mean Grade:        {mean_grade:.2f}")
            print(f"    Median Grade:      {median_grade:.2f}")

    # Print comparison table
    print("\n" + "=" * 110)
    print(f"METRICS SUMMARY - {set_name}")
    print("(Forecast ratings vs ground truth, graded activities only)")
    print("=" * 110)

    # Header
    print(
        f"{'Method':<35} {'N':>5} {'R^2':>8} {'PairOrd':>9} {'RMSE':>8} {'MAE':>8} {'SideAcc@3.5':>12} {'MeanGrade':>10} {'MedianGrade':>12}"
    )
    print("-" * 110)

    # Rows
    for name in [c["name"] for c in enabled_configs if c["enabled"]]:
        if name not in results:
            continue

        metrics = results[name]["metrics"]
        mean_grade = results[name]["mean_grade"]
        median_grade = results[name]["median_grade"]
        pairwise_val = results[name].get("pairwise", float("nan"))
        print(
            f"{name:<35} {metrics['n']:>5} {metrics['r2']:>8.4f} {pairwise_val:>9.4f} {metrics['rmse']:>8.4f} "
            f"{metrics['mae']:>8.4f} {metrics['side_accuracy_3_5']:>12.4f} "
            f"{mean_grade:>10.2f} {median_grade:>12.2f}"
        )

    # Generate and print LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    latex_table = generate_latex_table(results, set_name, enabled_configs)
    print(latex_table)
    print("=" * 80)

    print("\nComparison complete")

    return results, len(strict_ids)


def analyze_variance_across_runs(
    set_name: str,
    forecast_configs: list[dict[str, Any]],
    ground_truth_ratings: pd.Series,
    data_dir: Path,
) -> None:
    """Analyze rating/metric variance across multiple runs. Does not perform grading."""
    print("\n" + "=" * 80)
    print(f"VARIANCE ANALYSIS: {set_name}")
    print("=" * 80)

    enabled_configs = [c for c in forecast_configs if c["enabled"]]
    print(f"Number of runs to analyze: {len(enabled_configs)}")

    all_ratings = {}  # {run_name: Series of ratings}
    all_activity_ids = set()

    for config in enabled_configs:
        name = config["name"]
        filename = config["file"]
        forecast_path = data_dir / filename

        print(f"\n--- Loading: {name} ---")

        if not forecast_path.exists():
            print(f"WARNING: File not found: {forecast_path}")
            continue

        # Extract numeric ratings
        ratings = extract_forecast_ratings(
            forecast_path, parser=parse_last_line_label_after_forecast
        )
        print(f"  Extracted {len(ratings)} ratings")

        all_ratings[name] = ratings
        all_activity_ids.update(ratings.index)

    if not all_ratings:
        print("ERROR: No ratings loaded!")
        return

    common_ids = set.intersection(
        *[set(ratings.index) for ratings in all_ratings.values()]
    )
    common_ids = common_ids & set(ground_truth_ratings.index)

    print(f"\n{'=' * 80}")
    print("COMMON ACTIVITIES:")
    print(f"  Total unique activity IDs across all runs: {len(all_activity_ids)}")
    print(f"  Activities present in ALL runs AND ground truth: {len(common_ids)}")
    print(f"{'=' * 80}\n")

    if len(common_ids) < 10:
        print(
            f"WARNING: Very few common activities ({len(common_ids)}). Results may not be reliable."
        )

    run_metrics = {}  # {run_name: metrics_dict}

    for name, ratings in all_ratings.items():
        y_pred = ratings[list(common_ids)].astype(float).to_numpy()
        y_true = ground_truth_ratings[list(common_ids)].astype(float).to_numpy()

        metrics = calculate_metrics(y_true, y_pred)
        metrics["pairwise"] = pairwise_ordering_prob_excl_ties(y_true, y_pred)
        run_metrics[name] = metrics

    print("\nCalculating rating variance across runs for each activity...")
    activity_rating_variance = {}

    for aid in sorted(common_ids):
        ratings_for_activity = [all_ratings[name][aid] for name in all_ratings.keys()]
        activity_rating_variance[aid] = {
            "mean": np.mean(ratings_for_activity),
            "std": np.std(ratings_for_activity),
            "min": np.min(ratings_for_activity),
            "max": np.max(ratings_for_activity),
            "range": np.max(ratings_for_activity) - np.min(ratings_for_activity),
        }

    all_stds = [v["std"] for v in activity_rating_variance.values()]
    all_ranges = [v["range"] for v in activity_rating_variance.values()]

    print(f"\nRating Variance Across Runs (n={len(common_ids)} activities):")
    print(f"  Mean std dev per activity:     {np.mean(all_stds):.4f}")
    print(f"  Median std dev per activity:   {np.median(all_stds):.4f}")
    print(f"  Max std dev per activity:      {np.max(all_stds):.4f}")
    print(f"  Mean range per activity:       {np.mean(all_ranges):.4f}")
    print(f"  Median range per activity:     {np.median(all_ranges):.4f}")
    print(f"  Max range per activity:        {np.max(all_ranges):.4f}")

    metric_names = ["r2", "pairwise", "rmse", "mae", "side_accuracy_3_5"]
    metric_stats = {}

    for metric_name in metric_names:
        values = [run_metrics[name][metric_name] for name in all_ratings.keys()]
        metric_stats[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "values": values,
        }

    # Calculate skill if available (might need to be defined)
    # Assuming skill is R^2 - for now, just use R^2
    metric_stats["skill"] = metric_stats["r2"]

    print("\n" + "=" * 110)
    print(f"METRICS PER RUN - {set_name}")
    print(f"(n={len(common_ids)} common activities)")
    print("=" * 110)

    print(
        f"{'Run':<25} {'R^2':>10} {'PairOrd':>10} {'RMSE':>10} {'MAE':>10} {'SideAcc@3.5':>15} {'Skill':>10}"
    )
    print("-" * 110)

    for name in all_ratings.keys():
        metrics = run_metrics[name]
        print(
            f"{name:<25} {metrics['r2']:>10.4f} {metrics.get('pairwise', float('nan')):>10.4f} {metrics['rmse']:>10.4f} "
            f"{metrics['mae']:>10.4f} {metrics['side_accuracy_3_5']:>15.4f} "
            f"{metrics['r2']:>10.4f}"
        )  # Using R^2 as skill

    print("-" * 110)

    print(
        f"{'MEAN':<25} {metric_stats['r2']['mean']:>10.4f} "
        f"{metric_stats['pairwise']['mean']:>10.4f} "
        f"{metric_stats['rmse']['mean']:>10.4f} {metric_stats['mae']['mean']:>10.4f} "
        f"{metric_stats['side_accuracy_3_5']['mean']:>15.4f} "
        f"{metric_stats['skill']['mean']:>10.4f}"
    )

    print(
        f"{'STD DEV':<25} {metric_stats['r2']['std']:>10.4f} "
        f"{metric_stats['pairwise']['std']:>10.4f} "
        f"{metric_stats['rmse']['std']:>10.4f} {metric_stats['mae']['std']:>10.4f} "
        f"{metric_stats['side_accuracy_3_5']['std']:>15.4f} "
        f"{metric_stats['skill']['std']:>10.4f}"
    )

    print(
        f"{'MIN':<25} {metric_stats['r2']['min']:>10.4f} "
        f"{metric_stats['pairwise']['min']:>10.4f} "
        f"{metric_stats['rmse']['min']:>10.4f} {metric_stats['mae']['min']:>10.4f} "
        f"{metric_stats['side_accuracy_3_5']['min']:>15.4f} "
        f"{metric_stats['skill']['min']:>10.4f}"
    )

    print(
        f"{'MAX':<25} {metric_stats['r2']['max']:>10.4f} "
        f"{metric_stats['pairwise']['max']:>10.4f} "
        f"{metric_stats['rmse']['max']:>10.4f} {metric_stats['mae']['max']:>10.4f} "
        f"{metric_stats['side_accuracy_3_5']['max']:>15.4f} "
        f"{metric_stats['skill']['max']:>10.4f}"
    )

    print(
        f"{'RANGE':<25} "
        f"{metric_stats['r2']['max'] - metric_stats['r2']['min']:>10.4f} "
        f"{metric_stats['pairwise']['max'] - metric_stats['pairwise']['min']:>10.4f} "
        f"{metric_stats['rmse']['max'] - metric_stats['rmse']['min']:>10.4f} "
        f"{metric_stats['mae']['max'] - metric_stats['mae']['min']:>10.4f} "
        f"{metric_stats['side_accuracy_3_5']['max'] - metric_stats['side_accuracy_3_5']['min']:>15.4f} "
        f"{metric_stats['skill']['max'] - metric_stats['skill']['min']:>10.4f}"
    )

    print("=" * 110)

    print("\nCoefficient of Variation (CV = std/mean) across runs:")
    for metric_name in ["r2", "pairwise", "rmse", "mae", "side_accuracy_3_5", "skill"]:
        mean_val = metric_stats[metric_name]["mean"]
        std_val = metric_stats[metric_name]["std"]
        cv = (std_val / mean_val * 100) if mean_val != 0 else float("inf")
        print(f"  {metric_name:20s}: {cv:>8.2f}%")

    print("\nTop 10 most variable activities (by rating std dev):")
    sorted_activities = sorted(
        activity_rating_variance.items(), key=lambda x: x[1]["std"], reverse=True
    )[:10]

    print(
        f"{'Activity ID':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Range':>8}"
    )
    print("-" * 80)
    for aid, stats in sorted_activities:
        print(
            f"{aid:<30} {stats['mean']:>8.2f} {stats['std']:>8.4f} "
            f"{stats['min']:>8.2f} {stats['max']:>8.2f} {stats['range']:>8.2f}"
        )

    print("\nVariance analysis complete")


def print_test_set_wg_pairwise(
    data_dir: Path,
    ground_truth_ratings: pd.Series,
    org_labels: pd.Series,
    split_csv: Path = Path("../../data/train_val_test_ids.csv"),
    min_test_coverage: float = 0.5,
) -> None:
    """
    For any forecast file in data_dir that covers >= min_test_coverage fraction
    of the held-out test set, compute and print WG pairwise on those test activities.
    """
    if not split_csv.exists():
        print(f"WARNING: split file not found: {split_csv}")
        return

    split_df = pd.read_csv(split_csv)
    test_ids = set(split_df[split_df["split"] == "test"]["activity_id"])
    n_test = len(test_ids)
    threshold = int(n_test * min_test_coverage)

    print("\n" + "=" * 80)
    print(
        f"TEST SET WG PAIRWISE (files covering >= {min_test_coverage:.0%} of {n_test} test activities)"
    )
    print("=" * 80)

    # Only scan files following the forecast output naming convention
    forecast_globs = [
        "outputs_exactly_like_halawi_et_al_*.jsonl",
        "outputs_onlysummary_*.jsonl",
    ]

    def _skip_path(p):
        name = p.name
        return "_s1_" in name or "_s2_" in name or "second_try" in name

    found_any = False
    for glob_pat in forecast_globs:
        for forecast_path in sorted(data_dir.glob(glob_pat)):
            if _skip_path(forecast_path):
                continue
            ratings_series = extract_forecast_ratings(
                forecast_path, parser=parse_last_line_label_after_forecast
            )

            common = sorted(
                set(ratings_series.index) & set(ground_truth_ratings.index) & test_ids
            )
            if len(common) < threshold:
                continue

            found_any = True
            y_pred = ratings_series[common].astype(float).to_numpy()
            y_true = ground_truth_ratings[common].astype(float).to_numpy()
            groups = np.array([org_labels.get(aid, "unknown") for aid in common])

            _r_wg = within_group_pairwise_ordering_prob(y_true, y_pred, groups)
            wg_prob, n_pairs, n_groups = (
                _r_wg["prob"],
                _r_wg["n_pairs"],
                _r_wg["n_groups"],
            )

            print(f"\n  {forecast_path.name}")
            print(f"    Test activities covered:  {len(common)}/{n_test}")
            print(
                f"    WG pairwise (excl ties):  {wg_prob:.4f}  ({n_pairs} pairs, {n_groups} groups)"
            )

    if not found_any:
        print(
            f"  No forecast files with >= {threshold} test activities found in {data_dir}"
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    print("\nComparing LLM Forecast Outputs")
    print("=" * 80)

    execpool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="genai")
    _program_start = datetime.now()

    try:
        # Create output directories
        GRADES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (GRADES_OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

        # Load ground truth ratings once
        print("\nLoading ground truth ratings...")
        if not RATINGS_JSONL.exists():
            print(f"ERROR: {RATINGS_JSONL} not found")
            return

        ground_truth = load_ground_truth_ratings(RATINGS_JSONL)
        print(f"Loaded {len(ground_truth)} ground truth ratings")

        # Load org labels for within-group pairwise
        print("Loading org labels...")
        _info_csv = Path(
            "../../data/info_for_activity_forecasting_old_transaction_types.csv"
        )
        _info_df = pd.read_csv(_info_csv, usecols=["activity_id", "reporting_orgs"])
        org_labels = _info_df.set_index("activity_id")["reporting_orgs"]
        print(f"  Loaded org labels for {len(org_labels)} activities")

        # Load val split IDs so ranking metrics are restricted to val only
        _split_csv = Path("../../data/train_val_test_ids.csv")
        _split_df = pd.read_csv(_split_csv)
        val_ids = set(_split_df[_split_df["split"] == "val"]["activity_id"])
        print(f"  Val split: {len(val_ids)} activities")

        print("\n" + "#" * 80)
        print("~300 activities (separate sets per forecast)")
        print("#" * 80)

        results_a, strict_count_a = compare_forecast_set(
            "set_a",
            FORECAST_CONFIGS["set1_300_samples"],
            ground_truth,
            OUTCOMES_JSONL,
            DATA_DIR,
            GRADES_OUTPUT_DIR,
            execpool,
            set_defining_file="outputs_exactly_like_halawi_et_al_rag_added_no_knn_no_rag_deepseek_minimal_val_s3_call_1.jsonl",
            org_labels=org_labels,
            restrict_ids=val_ids,
        )

        # Grade risk summaries and compute mean for human proxy vertical line
        print("\n" + "#" * 80)
        print("HUMAN PROXY: Grading risk summaries")
        print("#" * 80)
        RISKS_JSONL = Path("../../data/outputs_risks.jsonl")
        set_defining_path = (
            DATA_DIR
            / "outputs_exactly_like_halawi_et_al_rag_added_no_knn_no_rag_deepseek_minimal_val_s3_call_1.jsonl"
        )
        set_ids_a = set(load_jsonl_by_activity_id(set_defining_path).keys())
        risks = load_risks_as_forecasts(RISKS_JSONL, set_ids_a)
        print(f"  Loaded {len(risks)} risk summaries")

        risks_grades_path = GRADES_OUTPUT_DIR / "grades_outputs_risks.jsonl"
        grade_risks(risks, OUTCOMES_JSONL, risks_grades_path, execpool)

        risks_grades_numeric = load_grades_with_ids(risks_grades_path)
        risks_grades_filtered = {
            aid: g for aid, g in risks_grades_numeric.items() if aid in set_ids_a
        }
        human_proxy_mean_grade = (
            float(np.mean(list(risks_grades_filtered.values())))
            if risks_grades_filtered
            else None
        )
        print(
            f"  Human proxy mean grade: {human_proxy_mean_grade:.2f} (n={len(risks_grades_filtered)})"
        )

        # 2-pane comparison plot
        print("\nCreating 2-pane comparison plot...")
        plot_2pane_comparison(
            results_a,
            f"{strict_count_a} activities",
            GRADES_OUTPUT_DIR / "plots" / "2pane_mean_grade_vs_r2_and_pairwise.png",
            human_proxy_mean_grade=human_proxy_mean_grade,
        )

        # Process Set 3 (Variance Analysis - no grading)
        print("\n" + "#" * 80)
        print("SET 3: Variance Analysis - Fine-tuning Runs (100 samples)")
        print("#" * 80)

        analyze_variance_across_runs(
            "Set 3 - Fine-tuning Variance",
            FORECAST_CONFIGS["set3_variance_analysis"],
            ground_truth,
            DATA_DIR,
        )

        # Test-set WG pairwise for any files covering the full test set
        print_test_set_wg_pairwise(DATA_DIR, ground_truth, org_labels)

        print("\n" + "=" * 80)
        print("All comparisons complete!")
        print(f"Output directory: {GRADES_OUTPUT_DIR}")
        print(
            f"Completed in {(datetime.now() - _program_start).total_seconds():.2f}s\n"
        )
        _fig2_path = (
            GRADES_OUTPUT_DIR / "plots" / "2pane_mean_grade_vs_r2_and_pairwise.png"
        ).resolve()
        print(f"Figure 2 (2-pane plot) saved to: {_fig2_path}")

    finally:
        execpool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-show", action="store_true", help="Save plots without displaying them"
    )
    args = parser.parse_args()
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    main()
