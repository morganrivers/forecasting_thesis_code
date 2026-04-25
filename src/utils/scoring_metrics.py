"""
Unified scoring rules / metrics for regression and classification evaluation.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
from scipy.stats import kendalltau

import pandas as pd

# ---------------------------------------------------------------------------
# Regression Metrics
# ---------------------------------------------------------------------------


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    """R-squared (coefficient of determination). Wraps sklearn."""
    return float(r2_score(y_true, y_pred))


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """
    Adjusted R^2: 1 - (1-R^2) * (n-1)/(n-p-1).
    Returns np.nan when r2 is None or the denominator is non-positive.
    """
    if r2 is None or n - p - 1 <= 0:
        return np.nan
    return 1.0 - (1.0 - float(r2)) * (n - 1) / (n - p - 1)


# ---------------------------------------------------------------------------
# Accuracy Metrics
# ---------------------------------------------------------------------------


def true_hit_accuracy(y_true, y_pred) -> float:
    """
    Accuracy after rounding predictions to nearest integer.
    Compares round(y_pred) == round(y_true).
    """
    yt = np.rint(np.asarray(y_true, dtype=float)).astype(int)
    yp = np.rint(np.asarray(y_pred, dtype=float)).astype(int)
    return float((yt == yp).mean())


def side_accuracy(y_true, y_pred, threshold: float) -> float:
    """
    Binary side accuracy: fraction of samples where both y_true and y_pred
    are on the same side of the threshold.

    side = 1 if > threshold, 0 otherwise.

    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    side_true = y_true > threshold
    side_pred = y_pred > threshold

    return float(np.mean(side_true == side_pred))


# ---------------------------------------------------------------------------
# Correlation / Ranking Metrics
# ---------------------------------------------------------------------------


def spearman_correlation(y_true, y_pred) -> float:
    """
    Spearman rank correlation coefficient between true and predicted values.
    Returns the correlation coefficient (range: -1 to 1, higher is better).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


def brier_skill_score(y_true, y_pred, train_base_rate: float | None = None) -> float:
    """
    Brier skill score for binary classification.

    Baseline Brier score = mean((p - y_i)^2) where p = train_base_rate if
    provided, else p = mean(y_true) on the eval set.

    Skill = 1 - brier / baseline.  Positive = better than predicting base rate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    p = train_base_rate if train_base_rate is not None else float(y_true.mean())
    base_brier = float(np.mean((p - y_true) ** 2))
    if base_brier < 1e-10:
        return float("nan")
    brier = float(np.mean((y_pred - y_true) ** 2))
    return 1.0 - brier / base_brier


def within_group_spearman_correlation(y_true, y_pred, groups) -> dict:
    """
    Spearman rank correlation within groups, weighted by group size.
    Groups with fewer than 2 finite observations are skipped.
    Returns dict with "correlation" (size-weighted mean rho) and "n_groups".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)

    weighted_sum = 0.0
    total_weight = 0
    n_groups = 0

    for g in np.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        valid = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[valid], yp[valid]
        if len(yt) < 2:
            continue
        rho = spearmanr(yt, yp).statistic
        if np.isfinite(rho):
            weighted_sum += rho * len(yt)
            total_weight += len(yt)
            n_groups += 1

    corr = weighted_sum / total_weight if total_weight > 0 else float("nan")
    return {"correlation": corr, "n_groups": n_groups}


def pairwise_ordering_prob_excl_ties(y_true, y_pred) -> float:
    """
    Probability that the model correctly orders a pair, given both true labels and
    predicted labels differ (ties in either are excluded).

    Denominator = pairs where y_true[i] != y_true[j] AND y_pred[i] != y_pred[j]
    Numerator   = pairs where sign(y_pred diff) == sign(y_true diff)

    Range: 0.0 (perfectly reversed) to 1.0 (perfect ordering), 0.5 = random.
    Returns np.nan if there are no non-tied pairs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dt = y_true[i_idx] - y_true[j_idx]
    dp = y_pred[i_idx] - y_pred[j_idx]
    mask = (dt != 0) & (dp != 0)
    dt, dp = dt[mask], dp[mask]
    if len(dt) == 0:
        return np.nan
    correct = np.sum(dt * dp > 0)
    return float(correct / len(dt))


# ---------------------------------------------------------------------------
# Within-group Pairwise Ordering (canonical stratified metric)
# ---------------------------------------------------------------------------


def within_group_pairwise_ordering_prob(y_true, y_pred, groups) -> dict:
    """
    Pairwise ordering probability (excl. true and pred ties) within groups.

    The canonical stratified ranking metric used throughout this project.
    Each group's POP is computed independently and groups are weighted by group size.
    Returns dict with "prob" (size-weighted mean POP), "n_pairs", "n_groups".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = np.asarray(groups)

    group_probs: list[float] = []
    group_sizes: list[int] = []
    n_pairs = 0

    for g in np.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        valid = np.isfinite(yt) & np.isfinite(yp)
        yt, yp = yt[valid], yp[valid]
        if len(yt) < 2:
            continue
        i_idx, j_idx = np.triu_indices(len(yt), k=1)
        dt = yt[i_idx] - yt[j_idx]
        dp = yp[i_idx] - yp[j_idx]
        nontied_true = dt != 0
        dt, dp = dt[nontied_true], dp[nontied_true]
        nontied_pred = dp != 0
        dt, dp = dt[nontied_pred], dp[nontied_pred]
        if len(dt) == 0:
            continue
        n_pairs += len(dt)
        concordant = int(np.sum(dt * dp > 0))
        group_probs.append(concordant / len(dt))
        group_sizes.append(len(yt))

    if not group_probs:
        return {"prob": float("nan"), "n_pairs": n_pairs, "n_groups": 0}
    weights = np.array(group_sizes, dtype=float)
    prob = float(np.average(group_probs, weights=weights))
    return {"prob": prob, "n_pairs": n_pairs, "n_groups": len(group_probs)}


def within_group_pairwise_ordering_prob_on_reference_pairs(
    y_true, y_pred, y_reference, groups
) -> dict:
    """
    Pairwise ordering probability restricted to pairs where y_reference also makes a non-tied
    prediction, stratified within groups. Returns dict with "prob", "n_pairs", "n_groups".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_reference = np.asarray(y_reference, dtype=float)
    groups = np.asarray(groups)

    group_probs: list[float] = []
    group_sizes: list[int] = []
    n_pairs = 0

    for g in np.unique(groups):
        mask = groups == g
        yt, yp, yr = y_true[mask], y_pred[mask], y_reference[mask]
        valid = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(yr)
        yt, yp, yr = yt[valid], yp[valid], yr[valid]
        if len(yt) < 2:
            continue
        i_idx, j_idx = np.triu_indices(len(yt), k=1)
        dt = yt[i_idx] - yt[j_idx]
        dp = yp[i_idx] - yp[j_idx]
        dr = yr[i_idx] - yr[j_idx]
        pair_mask = (dt != 0) & (dr != 0) & (dp != 0)
        dt, dp = dt[pair_mask], dp[pair_mask]
        if len(dt) == 0:
            continue
        n_pairs += len(dt)
        concordant = int(np.sum(dt * dp > 0))
        group_probs.append(concordant / len(dt))
        group_sizes.append(len(yt))

    if not group_probs:
        return {"prob": float("nan"), "n_pairs": n_pairs, "n_groups": 0}
    weights = np.array(group_sizes, dtype=float)
    prob = float(np.average(group_probs, weights=weights))
    return {"prob": prob, "n_pairs": n_pairs, "n_groups": len(group_probs)}


def org_year_pairwise_ordering_prob(
    y_true, y_pred, orgs, years, target_orgs=None
) -> tuple:
    """
    Within-org, within-year pairwise ordering probability.
    Returns (weighted_avg, per_org) where weighted_avg uses org+year combined group key
    and per_org is {org: (pop, n_pairs)}.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    orgs = np.asarray(orgs, dtype=str)
    years = np.asarray(years, dtype=str)

    if target_orgs is None:
        target_orgs = list(np.unique(orgs))

    # Overall: single call with org+year combined group key
    group_keys = np.char.add(np.char.add(orgs, "|||"), years)
    _r = within_group_pairwise_ordering_prob(y_true, y_pred, group_keys)
    weighted_avg = _r["prob"]

    # Per-org breakdown
    per_org: dict = {}
    for org in target_orgs:
        mask = orgs == org
        if mask.sum() < 2:
            per_org[org] = (np.nan, 0)
            continue
        _ro = within_group_pairwise_ordering_prob(
            y_true[mask], y_pred[mask], years[mask]
        )
        per_org[org] = (_ro["prob"], _ro["n_pairs"])

    return weighted_avg, per_org
