"""
Four generalizability analyses for the 14 curated outcome tag models,
mirroring E_overall_rating_extrapolate_scaling.py.

  A: Per-org POP heterogeneity -- printed table (uses saved predictions)
  B: Calibration reliability diagrams -- tall PNG (uses saved predictions)
  C: Learning curve -- 14 overlaid tag lines, POP + within-group POP
  D: Sliding window distance -- 14 overlaid tag lines, POP + within-group POP

A and B load from G_outcome_tag_train saved predictions; C and D retrain fresh.

EVAL_ON_VAL = True   # <- flip to False to evaluate on test set
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# -- Paths ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
UTILS_DIR = SRC_DIR / "utils"
DATA_DIR = SRC_DIR.parent / "data"
OUT_DIR = DATA_DIR / "outcome_tags"
PLOTS_DIR = OUT_DIR / "generalizability_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

for _p in [str(UTILS_DIR), str(SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -- Config --------------------------------------------------------------------
EVAL_ON_VAL = True  # default; overridden by --test flag at runtime
NOLIMITS = False  # overridden by --nolimits: no per-tag feature selection, no custom RF overrides
RUN_C = True  # learning curve

# Learning curve
N_LC_DROP_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_LC_SAMPLES = 3  # random subsamples per fraction to average
N_ESTIMATORS_CURVE = 100  # RF trees for C/D (speed); G_outcome_tag_train uses 500

# Sliding window (analysis D)
# When EVAL_ON_VAL=True:  pool = train_idx                eval = val_idx
# When EVAL_ON_VAL=False: pool = train_idx + val_idx      eval = test_idx
N_WINDOW_SIZE = 300  # fixed training window size (activities)
N_WINDOWS = 10  # number of windows to slide

# -- Imports from G_outcome_tag_train -----------------------------------------------
from G_outcome_tag_train import (
    ADD_START_YEAR_CORRECTION,
    APPLIED_TAGS,
    CORRECT_RF_BEFORE_ET,
    DROP_NOISY_FEATURE_GROUPS,
    KEEP_REPORTING_ORGS,
    LATEST_VALIDATION_POINT,
    MIN_TAG_TRAIN_COUNT_LOW,
    NOISY_FEATURE_GROUPS,
    OUT_REGULARIZATION,
    SKIP_START_YEAR_CORRECTION_TAGS,
    TAG_RF_PARAMS_OVERRIDES,
    TAGS_SKIP_FEATURE_SELECTION,
    apply_manual_factor_blend,
    apply_start_year_correction,
    build_feature_matrix,
    get_feature_cols,
    get_train_activity_ids_from_dates,
    load_applied_tags,
    load_per_tag_strategies,
    split_latest_by_date_with_cutoff,
    train_rf_et_ensemble,
)
from G_outcome_tag_train import (
    INFO_FOR_ACTIVITY_FORECASTING as INFO_CSV_PATH,
)
from H_outcome_tag_evaluate import (
    TAG_GROUPS,
)
from scoring_metrics import (
    pairwise_ordering_prob_excl_ties as pairwise_ordering_prob,
)
from scoring_metrics import (
    side_accuracy,
)
from scoring_metrics import (
    within_group_pairwise_ordering_prob as _wg_pop_fn,
)

# -- Curated tags --------------------------------------------------------------
# 14 tags from TAG_GROUPS (Finance, Rescoping, Target achievement)
CURATED_TAGS: list[str] = [
    tag for _group in TAG_GROUPS.values() for _lbl, tag in _group
]

TAG_SHORT: dict[str, str] = {
    tag: tag for _group in TAG_GROUPS.values() for _lbl, tag in _group
}
TAG_GROUP_MEMBERSHIP: dict[str, str] = {
    tag: gname for gname, items in TAG_GROUPS.items() for _lbl, tag in items
}

# One color per tag group; lines within a group share hue family
_GROUP_CMAPS = {
    "Finance & budget": plt.cm.Blues,
    "Activity Rescoping": plt.cm.Oranges,
    "Target achievement": plt.cm.Greens,
}


def _tag_colors() -> dict[str, tuple]:
    colors: dict[str, tuple] = {}
    for gname, items in TAG_GROUPS.items():
        cmap = _GROUP_CMAPS[gname]
        n = len(items)
        for i, (_lbl, tag) in enumerate(items):
            frac = 0.45 + 0.5 * (i / max(n - 1, 1))
            colors[tag] = cmap(frac)
    return colors


TAG_COLORS = _tag_colors()

ORG_SHORT: dict[str, str] = {
    "UK - Foreign, Commonwealth Development Office (FCDO)": "FCDO",
    "Asian Development Bank": "ADB",
    "World Bank": "WB",
    "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); Federal Ministry for Economic Cooperation and Development (BMZ)": "BMZ",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _within_group_pop(
    y_true_s: pd.Series,
    y_pred_s: pd.Series,
    group_key_s: pd.Series,
) -> float:
    """POP restricted to pairs sharing the same (reporting_org, start_year) group."""
    idx = y_true_s.index.intersection(y_pred_s.index).intersection(group_key_s.index)
    prob = _wg_pop_fn(
        y_true_s.reindex(idx).to_numpy(dtype=float),
        y_pred_s.reindex(idx).to_numpy(dtype=float),
        group_key_s.reindex(idx).to_numpy(),
    )["prob"]
    return float(prob)


# ---------------------------------------------------------------------------
# Build full feature matrix  (needed for C and D)
# ---------------------------------------------------------------------------


def _build_data() -> tuple:
    """
    Build feature matrix + merge tags. Returns:
      (data, train_idx, val_idx, test_idx, feature_cols, per_tag_strats)
    """
    print("\nComputing training activity IDs...")
    feat_cutoff = LATEST_VALIDATION_POINT if not EVAL_ON_VAL else None
    train_activity_ids = get_train_activity_ids_from_dates(
        str(INFO_CSV_PATH),
        cutoff_date=feat_cutoff,
        keep_orgs=KEEP_REPORTING_ORGS,
    )
    print(
        f"  {len(train_activity_ids)} training activities identified (cutoff={feat_cutoff})"
    )

    print("Building feature matrix (3-5 min) ...")
    data = build_feature_matrix(train_activity_ids=train_activity_ids)

    print("Loading and merging applied tags...")
    tags_df, model_cols = load_applied_tags(APPLIED_TAGS)
    data = data.join(tags_df, how="left")
    for col in model_cols:
        if col.endswith("_success") or col.endswith("_attempted"):
            if col not in data.columns:
                data[col] = np.nan
        else:
            if col not in data.columns:
                data[col] = 0
            else:
                data[col] = data[col].fillna(0).astype(int)

    # Ensure start_date and start_year are present
    from feature_engineering import pick_start_date

    if "start_date" not in data.columns:
        data["start_date"] = data.apply(pick_start_date, axis=1)
    data["start_year"] = data["start_date"].dt.year

    print(f"Data shape: {data.shape}")

    train_idx, val_idx, test_idx = split_latest_by_date_with_cutoff(data, "start_date")
    print(f"Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    feature_cols = get_feature_cols(data)
    if DROP_NOISY_FEATURE_GROUPS:
        noisy_set = set(NOISY_FEATURE_GROUPS)
        feature_cols = [c for c in feature_cols if c not in noisy_set]
    print(f"{len(feature_cols)} features after noisy-group drop")

    per_tag_strats = (
        {}
        if NOLIMITS
        else load_per_tag_strategies(OUT_REGULARIZATION, feature_cols, None)
    )

    return data, train_idx, val_idx, test_idx, feature_cols, per_tag_strats


# ---------------------------------------------------------------------------
# Training helpers for C, D, and E
# ---------------------------------------------------------------------------


def _active_cols_for_tag(
    col: str,
    model_type: str,
    feature_cols: list[str],
    per_tag_strats: dict,
) -> list[str]:
    """Return the feature subset to use for this tag, matching G_outcome_tag_train logic."""
    if col in TAGS_SKIP_FEATURE_SELECTION:
        return feature_cols
    feat_idx = per_tag_strats.get(col, {}).get("feat_idx")
    if feat_idx is not None:
        return [feature_cols[i] for i in feat_idx]
    return feature_cols


def _train_all_curated(
    data: pd.DataFrame,
    train_idx: pd.Index,
    eval_idx: pd.Index,
    feature_cols: list[str],
    per_tag_strats: dict,
    staged_model_types: dict[str, str],
    n_estimators: int = 100,
) -> dict[str, pd.Series]:
    """
    Train RF+ET (+ start-year correction + factor blend) for all curated tags,
    using the same model-type and feature-selection choices as G_outcome_tag_train.
    eval_idx is passed to apply_manual_factor_blend only for its internal logging.
    Returns {tag: probas_series indexed by data.index}.
    """
    import G_outcome_tag_train as _D

    # Temporarily override n_estimators for speed
    _orig_ne = _D.RF_PARAMS_BASE.get("n_estimators")
    _D.RF_PARAMS_BASE["n_estimators"] = n_estimators

    train_medians = data.loc[train_idx, feature_cols].median()
    all_results: list[dict] = []
    all_probas: dict[str, pd.Series] = {}

    for col in CURATED_TAGS:
        if col not in data.columns:
            continue

        y_train_col = data.loc[train_idx, col].dropna()
        n_train_pos = int(y_train_col.sum())
        n_tr = len(y_train_col)

        if n_train_pos < MIN_TAG_TRAIN_COUNT_LOW:
            all_results.append(
                {
                    "tag": col,
                    "model_type": "skip",
                    "train_n": n_tr,
                    "train_n_pos": n_train_pos,
                    "train_base_rate": n_train_pos / max(n_tr, 1),
                }
            )
            continue

        model_type = staged_model_types.get(col, "rf+ET")
        # Normalise variant names
        if model_type in ("rf", "rf+ET", "rf+ridge"):
            model_type = "rf+ET"
        if model_type == "const_base":
            continue  # no model to train; excluded from C/D plots

        active_cols = _active_cols_for_tag(
            col, model_type, feature_cols, per_tag_strats
        )
        X_tr = data.loc[y_train_col.index, active_cols].astype(float)
        X_all = data[active_cols].astype(float).fillna(train_medians[active_cols])

        try:
            rf_p, et_p, _, _ = train_rf_et_ensemble(
                X_tr,
                y_train_col,
                X_all,
                n_train_pos=n_train_pos,
                rf_params_override=TAG_RF_PARAMS_OVERRIDES.get(col),
            )
            if (
                CORRECT_RF_BEFORE_ET
                and ADD_START_YEAR_CORRECTION
                and col not in SKIP_START_YEAR_CORRECTION_TAGS
            ):
                corrected_rf, _ = apply_start_year_correction(
                    rf_p,
                    data.index,
                    data["start_date"],
                    train_idx,
                    y_train_col,
                )
                ens_p = (corrected_rf + et_p) / 2.0
            else:
                ens_p = (rf_p + et_p) / 2.0
        except Exception as e:
            print(f"  [{col}] training error ({model_type}): {e}")
            continue

        if (
            ADD_START_YEAR_CORRECTION
            and col not in SKIP_START_YEAR_CORRECTION_TAGS
            and not CORRECT_RF_BEFORE_ET
        ):
            ens_p, _ = apply_start_year_correction(
                ens_p,
                data.index,
                data["start_date"],
                train_idx,
                y_train_col,
            )

        col_sfx = "rf"
        all_probas[f"{col}__{col_sfx}"] = pd.Series(ens_p, index=data.index)
        all_results.append(
            {
                "tag": col,
                "model_type": model_type,
                "train_n": n_tr,
                "train_n_pos": n_train_pos,
                "train_base_rate": n_train_pos / max(n_tr, 1),
            }
        )

    # Restore n_estimators
    _D.RF_PARAMS_BASE["n_estimators"] = _orig_ne

    # Apply factor blending (post-processes all_probas and all_results in-place)
    apply_manual_factor_blend(
        data,
        train_idx,
        eval_idx,
        feature_cols,
        train_medians,
        all_results,
        all_probas,
    )

    # Extract final proba series per curated tag
    out: dict[str, pd.Series] = {}
    for col in CURATED_TAGS:
        for sfx in ("__rf", "__const_base"):
            key = f"{col}{sfx}"
            if key in all_probas:
                out[col] = all_probas[key]
                break
    return out


def _eval_tag_metrics(
    tag: str,
    probas_s: pd.Series,
    data: pd.DataFrame,
    eval_idx: pd.Index,
    train_pos_rate: float | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Return (pop, wg_pop, acc, acc_ratio, majority_acc) for tag on eval_idx.

    acc          -- model accuracy at threshold 0.5
    majority_acc -- accuracy of always-predict-majority-class baseline, evaluated on val set
                   (majority class determined by train_pos_rate, accuracy measured on eval set)
                   Uses train_pos_rate if provided (consistent with G_outcome_tag_train), else eval prevalence.
    acc_ratio    -- acc / majority_acc  (1.0 = same as baseline, > 1.0 = better)
    """
    NAN5 = (float("nan"),) * 5

    if tag not in data.columns:
        return NAN5

    y_true_s = data.loc[eval_idx, tag]
    y_pred_s = probas_s.reindex(eval_idx)
    mask = y_true_s.notna() & y_pred_s.notna()

    yt = y_true_s[mask].to_numpy(dtype=float)
    yp = y_pred_s[mask].to_numpy(dtype=float)

    if len(yt) < 2 or yt.sum() == 0 or yt.sum() == len(yt):
        return NAN5

    try:
        pop = float(pairwise_ordering_prob(yt, yp))
    except Exception:
        pop = float("nan")

    group_key = (
        data.loc[eval_idx, "reporting_orgs"].fillna("unknown").astype(str)
        + "|||"
        + data.loc[eval_idx, "start_year"].fillna(-1).astype(int).astype(str)
    )
    wg = _within_group_pop(y_true_s, y_pred_s, group_key)

    val_pos_rate = float(yt.mean())
    tr_rate = train_pos_rate if train_pos_rate is not None else val_pos_rate
    # Baseline: always predict the majority class from training, evaluated on val set.
    majority_acc = val_pos_rate if tr_rate >= 0.5 else (1.0 - val_pos_rate)
    acc = side_accuracy(yt, yp, 0.5)
    acc_ratio = acc / majority_acc if majority_acc > 0 else float("nan")

    return pop, wg, acc, acc_ratio, majority_acc


def _add_group_legend(fig: plt.Figure) -> None:
    """Add tag-group colour patches as a figure-level legend."""
    import matplotlib.patches as mpatches

    patches = [mpatches.Patch(color=_GROUP_CMAPS[g](0.70), label=g) for g in TAG_GROUPS]
    fig.legend(
        handles=patches,
        loc="upper center",
        ncol=3,
        fontsize=9,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 1.01),
    )


def _plot_extrapolated(
    df_rows: list[dict],
    eval_label: str,
    out_path: Path,
    majority_acc_by_tag: dict[str, float],
    n_extrap: int = 5000,
) -> None:
    """
    Fit an exponential saturation curve to each tag's learning data and extrapolate.
    Uses f(n) = L - b*exp(-k*n), which correctly starts from a non-zero baseline.
    Projections are printed to terminal; no labels on the plot.
    """
    from scipy.optimize import curve_fit

    def _sat(n, L, b, k):
        """Exponential saturation: starts at L-b, approaches L as n->inf."""
        return L - b * np.exp(-k * n)

    df = pd.DataFrame(df_rows)
    ns = df["n_train"].to_numpy(dtype=float)
    n_max = float(ns.max())

    xs_fit = np.linspace(0, n_extrap * 1.05, 300)

    fig, (ax_wg, ax_ar) = plt.subplots(1, 2, figsize=(18, 7))

    terminal_blocks: dict[str, list[str]] = {"wg_pop": [], "acc_ratio": []}

    for ax, metric_sfx, ylabel, title, add_hlines, term_key in [
        (
            ax_wg,
            "__wg_pop",
            "Within-group POP",
            f"WG-POP extrapolated to N={n_extrap}  ({eval_label})",
            False,
            "wg_pop",
        ),
        (
            ax_ar,
            "__acc_ratio",
            "Accuracy ratio (model / majority baseline)",
            f"Accuracy ratio extrapolated to N={n_extrap}  ({eval_label})",
            True,
            "acc_ratio",
        ),
    ]:
        all_obs_ys: list[np.ndarray] = (
            []
        )  # full-length (len(ns)) per tag, NaN where invalid
        proj_vals: list[float] = []  # y_proj per tag

        for tag in CURATED_TAGS:
            col_key = f"{tag}{metric_sfx}"
            if col_key not in df.columns:
                continue
            ys = df[col_key].to_numpy(dtype=float)

            # Clip accuracy ratio to max 2
            if add_hlines:
                ys = np.clip(ys, None, 2.0)

            valid = np.isfinite(ys) & np.isfinite(ns)
            if valid.sum() < 3:
                continue

            color = TAG_COLORS.get(tag, "grey")
            short = TAG_SHORT.get(tag, tag.replace("tag_", ""))

            y_obs = ys[valid]
            x_obs = ns[valid]

            # Observed data (no label)
            ax.scatter(x_obs, y_obs, color=color, s=20, alpha=0.6, zorder=4)
            ax.plot(x_obs, y_obs, color=color, linewidth=1.0, alpha=0.4, zorder=3)

            # Store full-length series for mean line (NaN at invalid positions)
            ys_full = np.where(valid, ys, np.nan)
            all_obs_ys.append(ys_full)

            # Exponential saturation fit: L - b*exp(-k*n)
            try:
                y_mean = float(y_obs.mean())
                y_max = float(y_obs.max())
                L0 = min(y_max * 1.1, 1.05 if not add_hlines else 2.0)
                b0 = max(L0 - y_mean, 0.01)
                k0 = 1.0 / float(x_obs.mean()) if x_obs.mean() > 0 else 0.001
                p0 = [L0, b0, k0]
                upper_L = 1.05 if not add_hlines else 2.0
                bounds = ([y_max, 0.0, 1e-6], [upper_L, upper_L, 1.0])
                popt, _ = curve_fit(
                    _sat, x_obs, y_obs, p0=p0, bounds=bounds, maxfev=8000
                )
                ys_fit = _sat(xs_fit, *popt)
                ax.plot(
                    xs_fit,
                    ys_fit,
                    color=color,
                    linewidth=1.8,
                    linestyle="-",
                    alpha=0.85,
                    zorder=5,
                )
                y_proj = float(_sat(n_extrap, *popt))
                ax.scatter(
                    [n_extrap],
                    [y_proj],
                    color=color,
                    marker="*",
                    s=160,
                    zorder=7,
                    edgecolors="black",
                    linewidths=0.4,
                )
                proj_vals.append(y_proj)
                terminal_blocks[term_key].append(f"  {short:<55s} {y_proj:.3f}")
            except Exception:
                ax.plot(
                    x_obs,
                    y_obs,
                    color=color,
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.5,
                    zorder=3,
                )
                terminal_blocks[term_key].append(f"  {short:<55s} (fit failed)")

            # Per-tag theoretical max horizontal line (acc ratio panel only)
            if add_hlines:
                maj = majority_acc_by_tag.get(tag, float("nan"))
                if np.isfinite(maj) and maj > 0:
                    theo_max = min(1.0 / maj, 2.0)
                    ax.axhline(
                        theo_max, color=color, linestyle="--", linewidth=0.8, alpha=0.45
                    )

        # -- Mean observed line (black, no fit) --------------------------------
        if all_obs_ys:
            mean_obs = np.nanmean(np.vstack(all_obs_ys), axis=0)
            valid_mean = np.isfinite(mean_obs)
            ax.plot(
                ns[valid_mean],
                mean_obs[valid_mean],
                color="black",
                linewidth=2.5,
                marker="o",
                markersize=5,
                zorder=9,
                alpha=0.95,
                label="mean",
            )

        # -- Mean projected star (black) ----------------------------------------
        if proj_vals:
            mean_proj = float(np.mean(proj_vals))
            ax.scatter(
                [n_extrap],
                [mean_proj],
                color="black",
                marker="*",
                s=300,
                zorder=10,
                edgecolors="black",
                linewidths=0.8,
                label=f"mean @ N={n_extrap}: {mean_proj:.3f}",
            )

        # Reference lines
        if not add_hlines:
            ax.axhline(1.0, color="black", linestyle=":", linewidth=0.9, alpha=0.5)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.6, alpha=0.3)
        ax.axvline(
            n_max,
            color="grey",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            label=f"Current N={int(n_max)}",
        )
        ax.axvline(
            n_extrap,
            color="tomato",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"N={n_extrap}",
        )
        if add_hlines:
            ax.set_ylim(top=2.05)

        ax.set_xlabel("Training set size (n)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
        ax.grid(True, alpha=0.3)

    _add_group_legend(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Plot saved -> {out_path}")

    # -- Terminal output --------------------------------------------------------
    sep = (
        "\n" * 10
        + "=" * 51
        + f"  EXTRAPOLATED PERFORMANCE (N={n_extrap})  "
        + "=" * 51
        + "\n"
    )
    print(sep)
    print(f"  {'Tag':<55s} WG-POP @ N={n_extrap}")
    print("  " + "-" * 70)
    for line in terminal_blocks["wg_pop"]:
        print(line)
    print()
    print(f"  {'Tag':<55s} Acc ratio @ N={n_extrap}")
    print("  " + "-" * 70)
    for line in terminal_blocks["acc_ratio"]:
        print(line)
    print("\n" * 10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--justplot",
        action="store_true",
        help="Re-plot C and D from cached JSON without recomputing",
    )
    parser.add_argument(
        "--justplot-calibration",
        dest="justplot_calibration",
        action="store_true",
        help="Re-plot B (calibration) from saved G_outcome_tag_train predictions only",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on test set (requires G_outcome_tag_train.py --use_test to have been run).",
    )
    parser.add_argument(
        "--nolimits",
        action="store_true",
        help="Use nolimits run outputs (requires G_outcome_tag_train.py --nolimits to have been run). "
        "Disables per-tag feature selection and TAG_RF_PARAMS_OVERRIDES for C/D/E retrain.",
    )
    parser.add_argument(
        "--slowcorrectextrapolate",
        action="store_true",
        help="Only run the learning curve and extrapolation plot using the correct "
        "n_estimators (from RF_PARAMS_BASE) and full RF+ET ensemble. "
        "Skips analyses A, B, D, E. Saves to *_slowcorrect_extrapolated.png.",
    )
    args = parser.parse_args()

    global EVAL_ON_VAL, PREDICTIONS_PATH, D_RESULTS_PATH
    global NOLIMITS, TAG_RF_PARAMS_OVERRIDES
    if args.test:
        EVAL_ON_VAL = False
        PREDICTIONS_PATH = OUT_DIR / "tag_predictions_test.csv"
        D_RESULTS_PATH = OUT_DIR / "tag_model_results_test.json"
        print("WARNING: TEST SET Run! Using test predictions.")
    if args.nolimits:
        NOLIMITS = True
        TAG_RF_PARAMS_OVERRIDES = {}
        PREDICTIONS_PATH = OUT_DIR / "tag_predictions_nolimits.csv"
        D_RESULTS_PATH = OUT_DIR / "tag_model_results_nolimits.json"
        print(
            "[nolimits] Using nolimits run outputs. Per-tag strategies and RF overrides disabled."
        )

    eval_label = "val" if EVAL_ON_VAL else "test"
    if args.nolimits:
        eval_label = f"{eval_label}_nolimits"

    if args.slowcorrectextrapolate:
        from G_outcome_tag_train import RF_PARAMS_BASE as _RF_PARAMS_BASE

        _correct_n_estimators = _RF_PARAMS_BASE["n_estimators"]
        slow_label = f"{eval_label}_slowcorrect"
        print(
            f"\n[slowcorrectextrapolate] n_estimators={_correct_n_estimators}, label={slow_label}"
        )
        data, train_idx, val_idx, test_idx, feature_cols, per_tag_strats = _build_data()
        eval_idx = val_idx if EVAL_ON_VAL else test_idx
        if not EVAL_ON_VAL:
            train_idx = train_idx.append(val_idx)
            print(
                f"[slowcorrectextrapolate] --test: merged val into train_idx -> n_train={len(train_idx)}"
            )

        staged_model_types: dict[str, str] = {}
        if D_RESULTS_PATH.exists():
            with D_RESULTS_PATH.open() as _f:
                for _r in json.load(_f):
                    staged_model_types[_r["tag"]] = _r.get("model_type", "rf+ET")
        else:
            print(
                f"WARNING: {D_RESULTS_PATH} not found -- defaulting all tags to rf+ET"
            )

        rng = np.random.default_rng(42)
        n_full = len(train_idx)
        rows: list[dict] = []
        majority_acc_by_tag: dict[str, float] = {}

        for frac in N_LC_DROP_FRACS:
            n_keep = int(round(n_full * (1.0 - frac)))
            print(
                f"\n  Drop {int(frac*100):>2}% -> n_train={n_keep}  ({N_LC_SAMPLES} samples) ...",
                flush=True,
            )
            s_pops: dict[str, list[float]] = {t: [] for t in CURATED_TAGS}
            s_wg_pops: dict[str, list[float]] = {t: [] for t in CURATED_TAGS}
            s_accs: dict[str, list[float]] = {t: [] for t in CURATED_TAGS}
            s_aratio: dict[str, list[float]] = {t: [] for t in CURATED_TAGS}
            s_n_trains: dict[str, list[int]] = {t: [] for t in CURATED_TAGS}

            for s_i in range(N_LC_SAMPLES):
                chosen = rng.choice(n_full, size=n_keep, replace=False)
                tr_idx = train_idx[np.sort(chosen)]
                for _tag in CURATED_TAGS:
                    if _tag in data.columns:
                        s_n_trains[_tag].append(
                            int(data.loc[tr_idx, _tag].dropna().__len__())
                        )
                probas = _train_all_curated(
                    data,
                    tr_idx,
                    eval_idx,
                    feature_cols,
                    per_tag_strats,
                    staged_model_types,
                    _correct_n_estimators,
                )
                for tag in CURATED_TAGS:
                    if tag not in probas:
                        continue
                    tr_pos_rate = (
                        float(data.loc[tr_idx, tag].dropna().mean())
                        if tag in data.columns
                        else None
                    )
                    pop, wg, acc, acc_ratio, maj_acc = _eval_tag_metrics(
                        tag,
                        probas[tag],
                        data,
                        eval_idx,
                        train_pos_rate=tr_pos_rate,
                    )
                    s_pops[tag].append(pop)
                    s_wg_pops[tag].append(wg)
                    s_accs[tag].append(acc)
                    s_aratio[tag].append(acc_ratio)
                    if tag not in majority_acc_by_tag and np.isfinite(maj_acc):
                        majority_acc_by_tag[tag] = maj_acc
                print(f"    sample {s_i+1}/{N_LC_SAMPLES} done", flush=True)
                if frac == 0.0 and s_i == 0:
                    import numpy as _np2

                    _eval_label_z = "test" if not EVAL_ON_VAL else "val"
                    _grp_key_z = (
                        data.loc[eval_idx, "reporting_orgs"]
                        .fillna("unknown")
                        .astype(str)
                        + "|||"
                        + data.loc[eval_idx, "start_year"]
                        .fillna(-1)
                        .astype(int)
                        .astype(str)
                    )
                    _org_counts_z = (
                        data.loc[eval_idx, "reporting_orgs"].value_counts().to_dict()
                    )
                    print(f"\n{'='*70}")
                    print(
                        "[WG-POP DEBUG] Z_tag_generalizability --slowcorrectextrapolate (frac=0, sample=0)"
                    )
                    print(f"  eval set          : {_eval_label_z}  n={len(eval_idx)}")
                    print(
                        f"  train set size    : {len(tr_idx)}  (full n_train={n_full} merged={not EVAL_ON_VAL})"
                    )
                    print(f"  n_estimators      : {_correct_n_estimators}")
                    print(f"  features          : {len(feature_cols)}")
                    print(f"  nolimits          : {NOLIMITS}")
                    print(
                        f"  group key         : reporting_orgs + start_year  ({_grp_key_z.nunique()} unique groups)"
                    )
                    print(
                        f"  org breakdown     : { {k[:30]: v for k, v in _org_counts_z.items()} }"
                    )
                    print(
                        f"  year range (eval) : {data.loc[eval_idx, 'start_year'].min():.0f}-{data.loc[eval_idx, 'start_year'].max():.0f}"
                    )
                    print(f"  {'tag':<52s}  {'in_probas':>9s}  {'wg_pop':>7s}")
                    print(f"  {'-'*73}")
                    _wg_vals_z = []
                    for _t in CURATED_TAGS:
                        _wg = s_wg_pops[_t][-1] if s_wg_pops[_t] else float("nan")
                        print(f"  {_t:<52s}  {str(_t in probas):>9s}  {_wg:7.3f}")
                        if _np2.isfinite(_wg):
                            _wg_vals_z.append(_wg)
                    print(f"  {'-'*73}")
                    print(
                        f"  {'mean (non-const)':<52s}  {'':>9s}  {_np2.nanmean(_wg_vals_z):7.3f}"
                    )
                    print(f"{'='*70}")

            row: dict = {"drop_pct": int(frac * 100), "n_train": n_keep}
            valid_pops, valid_wg, valid_ar = [], [], []
            for tag in CURATED_TAGS:

                def _mean_or_nan(lst):
                    vals = [v for v in lst if np.isfinite(v)]
                    return float(np.mean(vals)) if vals else float("nan")

                row[f"{tag}__pop"] = _mean_or_nan(s_pops[tag])
                row[f"{tag}__wg_pop"] = _mean_or_nan(s_wg_pops[tag])
                row[f"{tag}__acc"] = _mean_or_nan(s_accs[tag])
                row[f"{tag}__acc_ratio"] = _mean_or_nan(s_aratio[tag])
                row[f"{tag}__n_train"] = (
                    float(np.mean(s_n_trains[tag])) if s_n_trains[tag] else float("nan")
                )
                if np.isfinite(row[f"{tag}__pop"]):
                    valid_pops.append(row[f"{tag}__pop"])
                if np.isfinite(row[f"{tag}__wg_pop"]):
                    valid_wg.append(row[f"{tag}__wg_pop"])
                if np.isfinite(row[f"{tag}__acc_ratio"]):
                    valid_ar.append(row[f"{tag}__acc_ratio"])
            rows.append(row)
            print(
                f"    -> n={n_keep:4d}  "
                f"mean POP={np.mean(valid_pops):.3f}  "
                f"mean WG-POP={np.mean(valid_wg):.3f}  "
                f"mean acc_ratio={np.mean(valid_ar):.3f}"
                if (valid_pops and valid_wg and valid_ar)
                else f"    -> n={n_keep:4d}  (some metrics missing)"
            )

        _cache = PLOTS_DIR / f"learning_curve_rows_{slow_label}.json"
        with _cache.open("w") as _f:
            json.dump({"rows": rows, "majority_acc_by_tag": majority_acc_by_tag}, _f)

        _plot_extrapolated(
            rows,
            eval_label=slow_label,
            out_path=PLOTS_DIR / f"tag_learning_curve_{slow_label}_extrapolated.png",
            majority_acc_by_tag=majority_acc_by_tag,
            n_extrap=5000,
        )
        return

    if args.justplot_calibration:
        return

    if args.justplot:
        cache_C = PLOTS_DIR / f"learning_curve_rows_{eval_label}.json"
        if RUN_C:
            if not cache_C.exists():
                print(
                    f"ERROR: no cached data for C at {cache_C} -- run without --justplot first"
                )
            else:
                with cache_C.open() as f:
                    json.load(f)
        return

    if RUN_C:
        data, train_idx, val_idx, test_idx, feature_cols, per_tag_strats = _build_data()
        eval_idx = val_idx if EVAL_ON_VAL else test_idx

        # Load staged model types from G_outcome_tag_train saved results
        staged_model_types: dict[str, str] = {}
        if D_RESULTS_PATH.exists():
            with D_RESULTS_PATH.open() as _f:
                for _r in json.load(_f):
                    staged_model_types[_r["tag"]] = _r.get("model_type", "rf+ET")
        else:
            print(
                f"WARNING: {D_RESULTS_PATH} not found -- defaulting all tags to rf+ET"
            )


if __name__ == "__main__":
    main()
