"""
Split-stability SHAP analysis for outcome tags.
Always runs in nolimits mode (USE_PER_TAG_STRATEGY=False, TAG_RF_PARAMS_OVERRIDES={}).

Outputs (data/outcome_tags/):
  shap_split_stability_data_nolimits.pkl  -- read by J_outcome_tag_results_table.py
  shap_split_stability_{tag}_nolimits.png -- one plot per tag
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

SCRIPT_DIR = Path(__file__).resolve().parent
UTILS_DIR = SCRIPT_DIR.parent / "utils"
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
OUT_DIR = DATA_DIR / "outcome_tags"

for _p in [str(UTILS_DIR), str(SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import G_outcome_tag_train as D
from ml_models import DISABLE_ET

# -- Config --------------------------------------------------------------------
N_ESTIMATORS = 100  # trees per model (reduced for speed; 500 in G_outcome_tag_train)
TOP_K = 15  # features shown in each bar chart

SUFFIX = "_nolimits"
OUT_SPLIT_DATA = OUT_DIR / f"shap_split_stability_data{SUFFIX}.pkl"
N_SPLITS = 3  # number of equal training-data splits for the data-split stability plot
SPLIT_SEED = 42  # RNG seed for shuffling before splitting

LEAF = D.RF_PARAMS_BASE["min_samples_leaf"]


# -- Helpers -------------------------------------------------------------------


def get_target_tags() -> list[str]:
    return list(D.HARDCODED_14_TAGS)


def get_active_feat_cols(
    tag: str, per_tag_strats: dict, feature_cols: list[str]
) -> list[str]:
    strat = per_tag_strats.get(tag, {"strategy": "baseline", "feat_idx": None})
    feat_idx = strat.get("feat_idx")
    if feat_idx is None:
        return feature_cols
    cols = [feature_cols[i] for i in feat_idx if i < len(feature_cols)]
    return cols if cols else feature_cols


def train_rf_only(X_tr, y_tr, seed: int):
    """Train a single RF (no calibration) for SHAP, matching G_outcome_tag_train RF_PARAMS_BASE."""
    from sklearn.ensemble import RandomForestClassifier

    pos_rate = float(y_tr.mean())
    cw = None if pos_rate > 0.65 else "balanced"
    params = {
        **D.RF_PARAMS_BASE,
        "n_estimators": N_ESTIMATORS,
        "class_weight": cw,
        "random_state": seed,
    }
    params.pop("oob_score", None)
    rf = RandomForestClassifier(**params)
    rf.fit(X_tr, y_tr)
    return rf


def train_et_only(X_tr, y_tr, seed: int):
    """Train a single ExtraTrees (no calibration) for SHAP, matching G_outcome_tag_train ET derivation."""
    from sklearn.ensemble import ExtraTreesClassifier

    pos_rate = float(y_tr.mean())
    cw = None if pos_rate > 0.65 else "balanced"
    et = ExtraTreesClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=D.RF_PARAMS_BASE.get("max_depth", None),
        min_samples_leaf=D.RF_PARAMS_BASE["min_samples_leaf"],
        max_features=D.RF_PARAMS_BASE["max_features"],
        class_weight=cw,
        bootstrap=False,
        random_state=seed + 1,
        n_jobs=-1,
    )
    et.fit(X_tr, y_tr)
    return et


def shap_mean_abs(rf, X_tr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean |SHAP|, signed mean SHAP) per feature."""
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_tr)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    return np.abs(sv).mean(axis=0), sv.mean(axis=0)


def spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    r, _ = spearmanr(a, b)
    return float(r)


def plot_split_stability(
    tag: str,
    feat_cols: list[str],
    split_shap_means: list[np.ndarray],
    chosen_leaf: int,
    out_path: Path,
) -> None:
    """Bar chart: top-K features, one bar group per training split."""
    k = min(TOP_K, len(feat_cols))
    avg_mean = np.stack(split_shap_means).mean(axis=0)
    order = np.argsort(avg_mean)[::-1][:k]
    feat_lbls = [feat_cols[i][:30] for i in order]

    n = len(split_shap_means)
    x = np.arange(k)
    total_w = 0.75
    w = total_w / n
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for s, shap_mean in enumerate(split_shap_means):
        offset = (s - (n - 1) / 2) * w
        ax.barh(
            x + offset,
            shap_mean[order],
            w,
            color=colors[s % len(colors)],
            alpha=0.8,
            label=f"split {s + 1}",
        )

    ax.set_yticks(x)
    ax.set_yticklabels(feat_lbls, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(
        f"{tag}\nSHAP feature importance across {n} equal random training splits "
        f"(leaf={chosen_leaf}, top-{k} features by avg importance)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def split_stability_stats(
    feat_cols: list[str],
    split_abs_means: list[np.ndarray],
    split_signed_means: list[np.ndarray],
    top_k: int = 5,
) -> dict:
    """Compute top-K overlap, sign-flip rate, and rank correlation across splits."""
    n = len(split_abs_means)
    k = min(top_k, len(feat_cols))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    avg_abs = np.stack(split_abs_means).mean(axis=0)
    top_k_by_avg = set(np.argsort(avg_abs)[::-1][:k])
    per_split_topk = [set(np.argsort(m)[::-1][:k]) for m in split_abs_means]

    in_all = len(top_k_by_avg.intersection(*per_split_topk))
    avg_pairwise_overlap = float(
        np.mean([len(per_split_topk[i] & per_split_topk[j]) for i, j in pairs])
    )

    n_flips = sum(
        len({np.sign(sm[fi]) for sm in split_signed_means}) > 1 for fi in top_k_by_avg
    )

    rank_corrs = [
        spearman_rank_corr(split_abs_means[i], split_abs_means[j]) for i, j in pairs
    ]
    avg_rank_corr = float(np.mean(rank_corrs))

    return {
        "k": k,
        "in_all": in_all,
        "avg_pairwise_overlap": avg_pairwise_overlap,
        "sign_flips": n_flips,
        "avg_rank_corr": avg_rank_corr,
    }


def run_split_stability(
    data: pd.DataFrame,
    feat_cols: list[str],
    train_idx,
    per_tag_strats: dict,
    target_tags: list[str],
    train_medians: pd.Series,
    out_split_data: Path = OUT_SPLIT_DATA,
) -> None:
    print(f"\n{'='*80}")
    print(f"Split-stability analysis ({N_SPLITS} equal random training splits)")
    print(f"{'='*80}\n")

    rng = np.random.default_rng(SPLIT_SEED)
    shuffled = rng.permutation(train_idx.to_numpy())
    split_idxs = np.array_split(shuffled, N_SPLITS)

    all_stats: list[dict] = []
    split_tag_data: dict = {}

    for tag in target_tags:
        if tag not in data.columns:
            print(f"  [{tag}] not in data -- skipping")
            continue

        active_cols = get_active_feat_cols(tag, per_tag_strats, feat_cols)
        valid = data[tag].dropna().index

        split_abs_means: list[np.ndarray] = []
        split_signed_means: list[np.ndarray] = []
        skip = False

        for s, split_idx in enumerate(split_idxs):
            split_pd = pd.Index(split_idx)
            tr_valid = split_pd.intersection(valid)
            if len(tr_valid) < 10 or int(data.loc[tr_valid, tag].sum()) < 3:
                print(f"  [{tag}] split {s+1} too few examples -- skipping tag")
                skip = True
                break

            active_medians = train_medians[active_cols]
            X_sp = data.loc[tr_valid, active_cols].fillna(active_medians)
            y_sp = data.loc[tr_valid, tag].astype(int)
            print(
                f"  {tag}  split {s+1}/{N_SPLITS}  n={len(tr_valid)}  pos={int(y_sp.sum())}"
            )

            rf = train_rf_only(X_sp, y_sp, seed=SPLIT_SEED + s)
            abs_rf, signed_rf = shap_mean_abs(rf, X_sp)
            if DISABLE_ET:
                abs_mean = abs_rf
                signed_mean = signed_rf
            else:
                et = train_et_only(X_sp, y_sp, seed=SPLIT_SEED + s)
                abs_et, signed_et = shap_mean_abs(et, X_sp)
                abs_mean = (abs_rf + abs_et) / 2
                signed_mean = (signed_rf + signed_et) / 2
            split_abs_means.append(abs_mean)
            split_signed_means.append(signed_mean)

        if skip:
            continue

        tr_full = train_idx.intersection(valid)
        X_full = data.loc[tr_full, active_cols].fillna(train_medians[active_cols])
        y_full = data.loc[tr_full, tag].astype(int)
        print(f"  {tag}  full  n={len(tr_full)}  pos={int(y_full.sum())}")
        rf_full = train_rf_only(X_full, y_full, seed=SPLIT_SEED)
        abs_rf_f, signed_rf_f = shap_mean_abs(rf_full, X_full)
        if DISABLE_ET:
            full_abs_mean = abs_rf_f
            full_signed_mean = signed_rf_f
        else:
            et_full = train_et_only(X_full, y_full, seed=SPLIT_SEED)
            abs_et_f, signed_et_f = shap_mean_abs(et_full, X_full)
            full_abs_mean = (abs_rf_f + abs_et_f) / 2
            full_signed_mean = (signed_rf_f + signed_et_f) / 2

        split_tag_data[tag] = {
            "feature_cols": active_cols,
            "split_abs_means": [m.tolist() for m in split_abs_means],
            "split_signed_means": [m.tolist() for m in split_signed_means],
            "full_abs_mean": full_abs_mean.tolist(),
            "full_signed_mean": full_signed_mean.tolist(),
        }

        out_png = OUT_DIR / f"shap_split_stability_{tag}{SUFFIX}.png"
        plot_split_stability(tag, active_cols, split_abs_means, LEAF, out_png)
        stats = split_stability_stats(active_cols, split_abs_means, split_signed_means)
        all_stats.append(stats)

    if not all_stats:
        return
    k = all_stats[0]["k"]
    n_tags = len(all_stats)
    avg_in_all = np.mean([s["in_all"] for s in all_stats])
    avg_pairwise = np.mean([s["avg_pairwise_overlap"] for s in all_stats])
    total_sign_flips = sum(s["sign_flips"] for s in all_stats)
    avg_rank_corr = np.mean([s["avg_rank_corr"] for s in all_stats])

    print(f"\n{'='*60}")
    print(f"Split-stability summary across {n_tags} tags  ({N_SPLITS} splits, top {k})")
    print(f"{'='*60}")
    print(f"  Same top-{k} in ALL {N_SPLITS} splits (avg):  {avg_in_all:.1f} / {k}")
    print(f"  Avg pairwise top-{k} overlap:               {avg_pairwise:.1f} / {k}")
    print(
        f"  Sign flips in top-{k} (total across tags):  {total_sign_flips} / {n_tags * k}"
    )
    print(f"  Avg pairwise rank corr (full feature vec):  {avg_rank_corr:.3f}")
    print(f"{'='*60}")

    import pickle as _pkl

    with out_split_data.open("wb") as _f:
        _pkl.dump(split_tag_data, _f)
    print(f"\nSaved split SHAP data ({len(split_tag_data)} tags): {out_split_data}")


# -- Main ----------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SHAP split-stability analysis (nolimits, all activities)."
    )
    parser.parse_args()

    D.USE_PER_TAG_STRATEGY = False
    D.TAG_RF_PARAMS_OVERRIDES = {}
    print("[nolimits] USE_PER_TAG_STRATEGY=False, TAG_RF_PARAMS_OVERRIDES cleared.")
    print(f"Output paths suffixed with '{SUFFIX}'")
    print(
        f"n_estimators={N_ESTIMATORS} (reduced from G_outcome_tag_train default for speed)"
    )

    target_tags = get_target_tags()
    print(f"Target tags ({len(target_tags)}):")
    for t in target_tags:
        print(f"  {t}")
    print()

    print("Building feature matrix...", flush=True)
    data = D.build_feature_matrix()
    data = data[data["is_completed"].fillna(0).astype(int) == 1].copy()
    feat_cols = D.get_feature_cols(data)

    if D.DROP_NOISY_FEATURE_GROUPS:
        noisy = set(D.NOISY_FEATURE_GROUPS)
        feat_cols = [c for c in feat_cols if c not in noisy]

    train_idx, val_idx, test_idx = D.split_latest_by_date_with_cutoff(
        data, "start_date"
    )
    all_idx = train_idx.union(val_idx).union(test_idx)
    tag_df, tag_cols = D.load_applied_tags(D.APPLIED_TAGS)
    data = data.join(tag_df[tag_cols], how="left")
    print(
        f"Training on all activities: {len(all_idx)}  (train={len(train_idx)} + val={len(val_idx)} + test={len(test_idx)})"
    )
    print(f"Features: {len(feat_cols)}  leaf={LEAF}\n")

    all_medians = data.loc[all_idx, feat_cols].median()

    run_split_stability(
        data,
        feat_cols,
        all_idx,
        {},
        target_tags,
        all_medians,
        out_split_data=OUT_SPLIT_DATA,
    )


if __name__ == "__main__":
    main()
