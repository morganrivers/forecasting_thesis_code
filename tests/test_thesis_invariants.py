"""
Tests verifying that the implementation matches what the thesis documents.
"""

import numpy as np
import pandas as pd
import pytest

# -- Temporal splits (thesis §temporal_splits) --------------------------------


def test_split_dates_exact():
    """Thesis §temporal_splits: exact cutoff dates."""
    from split_constants import (
        LATEST_TRAIN_POINT,
        LATEST_VALIDATION_POINT,
        TOO_LATE_CUTOFF,
    )

    assert LATEST_TRAIN_POINT == "2013-02-06"
    assert LATEST_VALIDATION_POINT == "2016-06-06"
    assert TOO_LATE_CUTOFF == "2020-01-01"


def test_split_date_ordering():
    """Thesis §temporal_splits: train < val cutoff < test end — no overlap possible."""
    from split_constants import (
        LATEST_TRAIN_POINT,
        LATEST_VALIDATION_POINT,
        TOO_LATE_CUTOFF,
    )

    t = pd.to_datetime(LATEST_TRAIN_POINT)
    v = pd.to_datetime(LATEST_VALIDATION_POINT)
    c = pd.to_datetime(TOO_LATE_CUTOFF)
    assert t < v < c


def test_split_produces_disjoint_non_empty_sets():
    """Thesis §temporal_splits: split function returns three disjoint, non-empty sets."""
    from split_constants import split_latest_by_date_with_cutoff

    n = 100
    rng = np.random.default_rng(0)
    dates = pd.to_datetime(pd.date_range("2005-01-01", "2019-12-31", periods=n))
    df = pd.DataFrame({"start_date": dates}, index=range(n))
    train, val, test = split_latest_by_date_with_cutoff(df, "start_date")
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert len(train.intersection(val)) == 0
    assert len(train.intersection(test)) == 0
    assert len(val.intersection(test)) == 0


def test_split_cutoff_boundary_train():
    """Activities on exactly LATEST_TRAIN_POINT go into train (inclusive)."""
    from split_constants import (
        LATEST_TRAIN_POINT,
        split_latest_by_date_with_cutoff,
    )

    dates = [LATEST_TRAIN_POINT, "2014-01-01", "2017-01-01"]
    df = pd.DataFrame({"start_date": pd.to_datetime(dates)}, index=range(3))
    train, val, test = split_latest_by_date_with_cutoff(df, "start_date")
    assert 0 in train, "date == LATEST_TRAIN_POINT must be in train"


def test_split_cutoff_boundary_val():
    """Activities on exactly LATEST_VALIDATION_POINT go into val (inclusive)."""
    from split_constants import (
        LATEST_VALIDATION_POINT,
        split_latest_by_date_with_cutoff,
    )

    dates = ["2010-01-01", LATEST_VALIDATION_POINT, "2017-01-01"]
    df = pd.DataFrame({"start_date": pd.to_datetime(dates)}, index=range(3))
    train, val, test = split_latest_by_date_with_cutoff(df, "start_date")
    assert 1 in val, "date == LATEST_VALIDATION_POINT must be in val"


def test_split_excludes_too_late():
    """Activities on or after TOO_LATE_CUTOFF are excluded from all splits."""
    from split_constants import (
        TOO_LATE_CUTOFF,
        split_latest_by_date_with_cutoff,
    )

    dates = ["2010-01-01", "2014-01-01", "2017-01-01", TOO_LATE_CUTOFF, "2021-01-01"]
    df = pd.DataFrame({"start_date": pd.to_datetime(dates)}, index=range(5))
    train, val, test = split_latest_by_date_with_cutoff(df, "start_date")
    all_idx = set(train) | set(val) | set(test)
    assert 3 not in all_idx, "date == TOO_LATE_CUTOFF must be excluded"
    assert 4 not in all_idx, "date > TOO_LATE_CUTOFF must be excluded"


# -- Outcome tag list (thesis §outcome_tag_forecasting) -----------------------


def test_exactly_14_outcome_tags():
    """Thesis §outcome_tag_forecasting: exactly 14 binary outcome tags are forecasted."""
    from G_outcome_tag_train import HARDCODED_14_TAGS

    assert len(HARDCODED_14_TAGS) == 14
    assert len(set(HARDCODED_14_TAGS)) == 14, "Duplicate tag names in HARDCODED_14_TAGS"


def test_14_tags_match_thesis_list():
    """Thesis §outcome_tag_forecasting: each tag named in the bullet list is present."""
    from G_outcome_tag_train import HARDCODED_14_TAGS

    expected = {
        "tag_funds_cancelled_or_unutilized",
        "tag_funds_reallocated",
        "tag_high_disbursement",
        "tag_improved_financial_performance",
        "tag_project_restructured",
        "tag_closing_date_extended",
        "tag_targets_revised",
        "tag_policy_regulatory_reforms_success_success",
        "tag_targets_met_or_exceeded_success",
        "tag_over_budget_success",
        "tag_capacity_building_delivered_success",
        "tag_high_beneficiary_satisfaction_or_reach_success",
        "tag_gender_equitable_outcomes_success",
        "tag_private_sector_engagement_success",
    }
    assert set(HARDCODED_14_TAGS) == expected


def test_monitoring_tag_not_in_hardcoded_14():
    """Thesis §outcome_tag_forecasting: monitoring tag had poor val performance — not in the 14."""
    from G_outcome_tag_train import HARDCODED_14_TAGS

    assert "tag_monitoring_and_evaluation_challenges" not in HARDCODED_14_TAGS


# -- Noisy feature groups (thesis §feature_selection) -------------------------


def test_noisy_feature_group_counts():
    """Thesis §feature_selection: 7 governance (first 7) + 11 missingness (last 11) noisy features."""
    from G_outcome_tag_train import NOISY_FEATURE_GROUPS

    governance = NOISY_FEATURE_GROUPS[:7]
    missingness = NOISY_FEATURE_GROUPS[7:]
    assert len(governance) == 7, f"expected 7 governance features, got {governance}"
    assert (
        len(missingness) == 11
    ), f"expected 11 missingness features, got {missingness}"


def test_wgi_features_present():
    """All 5 WGI indicators must be in the governance slice (first 7 noisy features)."""
    from G_outcome_tag_train import NOISY_FEATURE_GROUPS

    governance = NOISY_FEATURE_GROUPS[:7]
    wgi = [f for f in governance if f.startswith("wgi_") and "missing" not in f]
    assert len(wgi) == 5, f"expected 5 WGI features in governance slice, got {wgi}"


def test_drop_noisy_flag_is_true():
    """Thesis §feature_selection: governance + missingness features are dropped (ablation result)."""
    from G_outcome_tag_train import DROP_NOISY_FEATURE_GROUPS

    assert (
        DROP_NOISY_FEATURE_GROUPS is True
    ), "thesis: ablation showed dropping noisy features improves performance — flag must be True"


# -- Tag factor blending (thesis §tag_group_avg) -------------------------------


def test_blend_constants():
    """Thesis §tag_group_avg: blend ramp from 100 to 400 minority-class training examples."""
    from G_outcome_tag_train import BLEND_LO, BLEND_HI

    assert BLEND_LO == 100
    assert BLEND_HI == 400


@pytest.mark.parametrize(
    "minority,expected",
    [
        (0, 0.0),  # below ramp: full group factor
        (100, 0.0),  # at BLEND_LO: still full group factor
        (250, 0.5),  # midpoint: 50/50
        (400, 1.0),  # at BLEND_HI: full per-tag model
        (500, 1.0),  # above ramp: still full per-tag model
    ],
)
def test_blend_formula(minority, expected):
    """Thesis §tag_group_avg: d_weight = clip((m-100)/(400-100), 0, 1)."""
    from G_outcome_tag_train import BLEND_LO, BLEND_HI

    d_weight = float(np.clip((minority - BLEND_LO) / (BLEND_HI - BLEND_LO), 0.0, 1.0))
    assert d_weight == pytest.approx(
        expected
    ), f"minority={minority}: expected d_weight={expected}, got {d_weight}"


# -- RF base configuration (thesis §RF_ET_ensemble) ---------------------------


def test_rf_base_params():
    """Thesis §RF_ET_ensemble: base config min_samples_leaf=5, max_features='sqrt', n_estimators=500."""
    from G_outcome_tag_train import RF_PARAMS_BASE

    assert RF_PARAMS_BASE["min_samples_leaf"] == 5
    assert RF_PARAMS_BASE["max_features"] == "sqrt"
    assert RF_PARAMS_BASE["n_estimators"] == 500
    assert RF_PARAMS_BASE.get("max_depth") is None


def test_per_tag_rf_overrides_count():
    """Thesis §RF_ET_ensemble: exactly 10 per-tag parameter overrides."""
    from G_outcome_tag_train import TAG_RF_PARAMS_OVERRIDES

    assert len(TAG_RF_PARAMS_OVERRIDES) == 10


def test_per_tag_rf_overrides_breakdown():
    """Thesis §RF_ET_ensemble: 5×leaf=20, 1×leaf=40, 1×leaf=10, 3×depth=10."""
    from G_outcome_tag_train import TAG_RF_PARAMS_OVERRIDES

    leaf_20 = [
        t for t, p in TAG_RF_PARAMS_OVERRIDES.items() if p.get("min_samples_leaf") == 20
    ]
    leaf_40 = [
        t for t, p in TAG_RF_PARAMS_OVERRIDES.items() if p.get("min_samples_leaf") == 40
    ]
    leaf_10 = [
        t for t, p in TAG_RF_PARAMS_OVERRIDES.items() if p.get("min_samples_leaf") == 10
    ]
    depth_10 = [
        t for t, p in TAG_RF_PARAMS_OVERRIDES.items() if p.get("max_depth") == 10
    ]
    assert len(leaf_20) == 5, f"thesis: 5 tags with min_samples_leaf=20, got {leaf_20}"
    assert len(leaf_40) == 1, f"thesis: 1 tag with min_samples_leaf=40, got {leaf_40}"
    assert len(leaf_10) == 1, f"thesis: 1 tag with min_samples_leaf=10, got {leaf_10}"
    assert len(depth_10) == 3, f"thesis: 3 tags with max_depth=10, got {depth_10}"


def test_class_weight_threshold():
    """Thesis §RF_ET_ensemble: threshold for disabling balanced class weight is 65%."""
    from G_outcome_tag_train import CLASS_WEIGHT_POS_RATE_THRESHOLD

    assert CLASS_WEIGHT_POS_RATE_THRESHOLD == 0.65


# -- Start-year trend correction (thesis §start_year_trend_correction) ---------


def test_start_year_ridge_alpha():
    """Thesis §start_year_trend_correction: ridge penalty α=50 shrinks slope toward zero."""
    from G_outcome_tag_train import START_YEAR_RIDGE_ALPHA

    assert START_YEAR_RIDGE_ALPHA == 50.0


def test_start_year_correction_clips_to_unit_interval():
    """Thesis §start_year_trend_correction: corrected probability is clipped to [0, 1]."""
    from G_outcome_tag_train import apply_start_year_correction

    n = 30
    idx = pd.RangeIndex(n)
    # Predictions all near 0.9; with a strong positive correction some would exceed 1.0
    y_pred = np.full(n, 0.9)
    start_dates = pd.Series(
        pd.date_range("2000-01-01", periods=n, freq="180D"), index=idx
    )
    # y_tr on first half: all 1 (early years) — yields negative residuals that would drag late years up
    tr_idx = idx[:15]
    y_tr = pd.Series(np.ones(15), index=tr_idx)
    corrected, _ = apply_start_year_correction(y_pred, idx, start_dates, tr_idx, y_tr)
    assert np.all(corrected >= 0.0), "corrected probas must not be negative"
    assert np.all(corrected <= 1.0), "corrected probas must not exceed 1.0"


def test_start_year_correction_on_constant_predictions():
    """Constant predictions with zero residuals → correction = 0 → output unchanged."""
    from G_outcome_tag_train import apply_start_year_correction

    n = 20
    idx = pd.RangeIndex(n)
    y_pred = np.full(n, 0.5)
    start_dates = pd.Series(
        pd.date_range("2005-01-01", periods=n, freq="90D"), index=idx
    )
    tr_idx = idx[:10]
    # y_tr equals the prediction → residuals are zero → correction should be ~0
    y_tr = pd.Series(np.full(10, 0.5), index=tr_idx)
    corrected, correction = apply_start_year_correction(
        y_pred, idx, start_dates, tr_idx, y_tr
    )
    assert np.allclose(
        corrected, 0.5, atol=1e-6
    ), "zero residuals should produce near-zero correction"


# -- Skip tags for correction (thesis §start_year_trend_correction) ------------


def test_skip_year_correction_tags():
    """Thesis §start_year_trend_correction: two tags where correction hurt val — must be skipped."""
    from G_outcome_tag_train import SKIP_START_YEAR_CORRECTION_TAGS

    assert "tag_monitoring_and_evaluation_challenges" in SKIP_START_YEAR_CORRECTION_TAGS
    assert "tag_high_disbursement" in SKIP_START_YEAR_CORRECTION_TAGS


# -- LLM adjustment ridge regression (thesis §LLM_adjustment) -----------------


def test_llm_corrector_alpha():
    """Thesis §LLM_adjustment: ℓ₂ penalty α=5."""
    from A_overall_rating_fit_and_evaluate import LLM_CORRECTOR_ALPHA

    assert LLM_CORRECTOR_ALPHA == 5.0


def test_llm_corrector_lambda():
    """Thesis §LLM_adjustment: scaling factor λ=1.0 (full correction applied)."""
    from A_overall_rating_fit_and_evaluate import LLM_CORRECTOR_LAM

    assert LLM_CORRECTOR_LAM == 1.0


def test_rating_clip_scale():
    """Rating scale is 0–5 (Highly Unsatisfactory=0 … Highly Satisfactory=5)."""
    from A_overall_rating_fit_and_evaluate import RATING_CLIP

    assert RATING_CLIP == (0.0, 5.0)


def test_llm_corrector_clip_range():
    """Thesis §LLM_adjustment: corrector clips to the same rating scale [0, 5]."""
    from A_overall_rating_fit_and_evaluate import LLM_CORRECTOR_CLIP, RATING_CLIP

    assert LLM_CORRECTOR_CLIP == RATING_CLIP


def test_llm_corrector_clips_output():
    """add_rf_llm_residual_corrector clips predictions to LLM_CORRECTOR_CLIP bounds."""
    import pandas as pd
    from A_overall_rating_fit_and_evaluate import (
        add_rf_llm_residual_corrector,
        LLM_CORRECTOR_CLIP,
    )

    n = 20
    rng = np.random.default_rng(42)
    idx = pd.RangeIndex(n)
    data = pd.DataFrame(
        {
            "pred_rf": rng.uniform(-1.0, 6.0, n),  # some outside [0, 5]
            "mean_all_llm_preds": rng.uniform(0.0, 5.0, n),
        },
        index=idx,
    )
    y = pd.Series(rng.uniform(1.0, 6.0, n), index=idx)
    _, _ = add_rf_llm_residual_corrector(
        data=data,
        y=y,
        meta_train_idx=idx[:10],
        alpha=5.0,
        lam=1.0,
        clip_lo=LLM_CORRECTOR_CLIP[0],
        clip_hi=LLM_CORRECTOR_CLIP[1],
    )
    assert data["pred_rf_llm_modded"].min() >= LLM_CORRECTOR_CLIP[0]
    assert data["pred_rf_llm_modded"].max() <= LLM_CORRECTOR_CLIP[1]


# -- Grade scale (thesis §grading_free_form) -----------------------------------


def test_grade_scale_thesis_values():
    """Thesis §grading_free_form: exact values stated — F=55, A+=97, D-=62, D=65, D+=68, C-=72."""
    from llm_grading_utils import GRADE_TO_PCT

    assert GRADE_TO_PCT["F"] == 55
    assert GRADE_TO_PCT["A+"] == 97
    assert GRADE_TO_PCT["D-"] == 62
    assert GRADE_TO_PCT["D"] == 65
    assert GRADE_TO_PCT["D+"] == 68
    assert GRADE_TO_PCT["C-"] == 72


def test_grade_scale_complete():
    """GRADE_TO_PCT covers all 13 letter grades in GRADE_ORDER."""
    from llm_grading_utils import GRADE_TO_PCT, GRADE_ORDER

    assert len(GRADE_ORDER) == 13
    assert set(GRADE_TO_PCT.keys()) == set(GRADE_ORDER)


def test_grade_scale_monotone_decreasing():
    """Grade percentages must decrease strictly from A+ down to F."""
    from llm_grading_utils import GRADE_TO_PCT, GRADE_ORDER

    pcts = [GRADE_TO_PCT[g] for g in GRADE_ORDER]
    for i in range(len(pcts) - 1):
        assert (
            pcts[i] > pcts[i + 1]
        ), f"{GRADE_ORDER[i]}={pcts[i]} must exceed {GRADE_ORDER[i+1]}={pcts[i+1]}"


# -- Brier skill score formula (thesis §brier_skill) --------------------------


def test_brier_skill_score_at_baseline():
    """Thesis §brier_skill: BSS=0 when predictions equal the base rate."""
    from scoring_metrics import brier_skill_score

    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=float)
    base_rate = float(y_true.mean())
    y_pred_baseline = np.full(len(y_true), base_rate)
    bss = brier_skill_score(y_true, y_pred_baseline, train_base_rate=base_rate)
    assert bss == pytest.approx(0.0, abs=1e-9)


def test_brier_skill_score_perfect():
    """Thesis §brier_skill: BSS=1 for perfect probabilistic predictions."""
    from scoring_metrics import brier_skill_score

    y_true = np.array([1, 0, 1, 0, 1], dtype=float)
    bss = brier_skill_score(y_true, y_true.copy(), train_base_rate=float(y_true.mean()))
    assert bss == pytest.approx(1.0, abs=1e-9)


def test_brier_skill_score_formula():
    """Thesis §brier_skill: BSS = 1 - BS/BS_ref — verify numerically."""
    from scoring_metrics import brier_skill_score

    y_true = np.array([1, 0, 1, 1, 0], dtype=float)
    y_pred = np.array([0.8, 0.3, 0.6, 0.9, 0.2])
    base_rate = 0.6
    bs = float(np.mean((y_pred - y_true) ** 2))
    bs_ref = float(np.mean((base_rate - y_true) ** 2))
    expected_bss = 1.0 - bs / bs_ref
    assert brier_skill_score(
        y_true, y_pred, train_base_rate=base_rate
    ) == pytest.approx(expected_bss)


def test_brier_skill_score_worse_than_baseline():
    """Predictions worse than the base rate give negative BSS."""
    from scoring_metrics import brier_skill_score

    y_true = np.array([1, 1, 1, 0, 0], dtype=float)
    y_pred = np.array([0.1, 0.1, 0.1, 0.9, 0.9])  # completely wrong
    bss = brier_skill_score(y_true, y_pred, train_base_rate=float(y_true.mean()))
    assert bss < 0


# -- Within-group pairwise ranking (thesis §scoring_metrics) ------------------


def test_pairwise_ranking_random_chance():
    """Thesis §scoring_metrics: random predictions give pairwise ranking ≈ 0.5."""
    from scoring_metrics import pairwise_ordering_prob_excl_ties as pop

    rng = np.random.default_rng(99)
    y_true = rng.integers(0, 2, 200).astype(float)
    y_pred = rng.uniform(0, 1, 200)
    result = pop(y_true, y_pred)
    assert 0.3 < result < 0.7, f"random predictions should give ~0.5, got {result}"


def test_pairwise_ranking_perfect():
    """Thesis §scoring_metrics: predictions perfectly ordered give pairwise ranking = 1.0."""
    from scoring_metrics import pairwise_ordering_prob_excl_ties as pop

    y_true = np.array([1, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    result = pop(y_true, y_pred)
    assert result == pytest.approx(1.0)


def test_pairwise_ranking_reverse():
    """Perfectly reversed ordering gives pairwise ranking = 0.0."""
    from scoring_metrics import pairwise_ordering_prob_excl_ties as pop

    y_true = np.array([1, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    result = pop(y_true, y_pred)
    assert result == pytest.approx(0.0)
