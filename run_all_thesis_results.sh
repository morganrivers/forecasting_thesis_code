#!/usr/bin/env bash
# Runs all scripts needed to reproduce thesis results and saves all output to a single text file.
# Usage: bash run_all_thesis_results.sh
# Output: thesis_results_output.txt in the repo root

set -uo pipefail

QUICK=false
for arg in "$@"; do
    if [ "$arg" = "--quick" ]; then
        QUICK=true
    fi
done

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$REPO_ROOT/thesis_results_output.txt"
SHAP_PNG="$REPO_ROOT/src/pipeline/feature_importances_plot.png"
FIGURE2_PNG="$REPO_ROOT/data/forecast_grades/plots/2pane_mean_grade_vs_r2_and_pairwise.png"

> "$OUT"

run_script() {
    local heading="$1"
    local what_to_look_for="$2"
    local dir="$3"
    local cmd="$4"

    {
        printf '%.0s=' {1..80}
        printf '\n'
        printf '%s\n' "$heading"
        printf '%s\n' "$what_to_look_for"
        printf '%.0s=' {1..80}
        printf '\n'
    } >> "$OUT"

    echo "Running: $cmd (in $dir)"
    (cd "$dir" && eval "$cmd") >> "$OUT" 2>&1
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "ERROR: '$cmd' (in $dir) exited with code $exit_code" >&2
        echo "See $OUT for details." >&2
        exit $exit_code
    fi

    # 10 blank lines between sections
    printf '\n%.0s' {1..10} >> "$OUT"
}

# ============================================================
# Overall rating prediction (Table 1, R^2, RMSE, MAE, pairwise)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/A_overall_rating_fit_and_evaluate.py" \
    "LOOK FOR: '=== RESULTS (terminal) ===' -- Pairwise=, R^2 lines per method. Also writes data/rating_model_outputs/ (predictions, pickle, feature importances). AND: Feature importance table printed after '=== RESULTS (terminal) ===' section. AND: Adjusted R^2 on the train+val set for all orgs gives \\RsqTraining" \
    "$REPO_ROOT/src/pipeline" \
    "python3 A_overall_rating_fit_and_evaluate.py importances"

# ============================================================
# SHAP beeswarm plot (Figure 3)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/B_overall_rating_plot_shap.py" \
    "LOOK FOR: Confirmation that feature_importances_plot.png was saved. Requires A_overall_rating_fit_and_evaluate.py to have been run first." \
    "$REPO_ROOT/src/pipeline" \
    "python3 B_overall_rating_plot_shap.py"

# ============================================================
# In-sample R^2 on training+validation set (\RsqTrainingOLS, \RsqAdjTrainingOLS)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/C_overall_rating_insample_r2.py" \
    "LOOK FOR: 'R^2' and 'Adjusted R^2' for OLS fit on train+val -- gives \\RsqTrainingOLS, \\RsqAdjTrainingOLS." \
    "$REPO_ROOT/src/pipeline" \
    "python3 C_overall_rating_insample_r2.py"

# ============================================================
# OOS val predictions (prerequisite for F_llm_score_forecast_narratives.py)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/D_overall_rating_generate_oos_predictions.py" \
    "LOOK FOR: Generates data/rf_oos_val_predictions.csv required by F_llm_score_forecast_narratives.py." \
    "$REPO_ROOT/src/pipeline" \
    "python3 D_overall_rating_generate_oos_predictions.py"

# ============================================================
# Extrapolated scaling: \PairwiseModelExtrapolated
# ============================================================
if [ "$QUICK" != "true" ]; then
run_script \
    "SCRIPT: src/pipeline/E_overall_rating_extrapolate_scaling.py" \
    "LOOK FOR: 'pop_proj_5x' from analysis_17_learning_curve -- gives \\PairwiseModelExtrapolated (log-linear extrapolation to 5x training data)." \
    "$REPO_ROOT/src/pipeline" \
    "python3 E_overall_rating_extrapolate_scaling.py --lc-pop"
fi

# ============================================================
# Narrative similarity grades (\BestNarrativeGrade, \RisksOnlyGrade)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/F_llm_score_forecast_narratives.py" \
    "LOOK FOR: 'Mean Grade: <value>' per forecast config. Best RF-forced full-context config -> \\BestNarrativeGrade; risks-only config -> \\RisksOnlyGrade." \
    "$REPO_ROOT/src/pipeline" \
    "python3 F_llm_score_forecast_narratives.py --no-show"

# ============================================================
# Outcome tag prediction -- test run calculates results prerequisite
# ============================================================
run_script \
    "SCRIPT: src/pipeline/G_outcome_tag_train.py --use_test (test run)" \
    "LOOK FOR: Nothing, prerequisite for next script" \
    "$REPO_ROOT/src/pipeline" \
    "python3 G_outcome_tag_train.py --use_test --hardcode_tags"

# ============================================================
# Outcome tag prediction -- test run calculates results prerequisite
# ============================================================
run_script \
    "SCRIPT: src/pipeline/G_outcome_tag_train.py --use_test (test run)" \
    "LOOK FOR: Nothing, prerequisite for next script" \
    "$REPO_ROOT/src/pipeline" \
    "python3 G_outcome_tag_train.py --use_test --nolimits --hardcode_tags"
    
# ============================================================
# Outcome tag prediction -- test run (per-tag feature-selection values)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/H_outcome_tag_evaluate.py --test (test run)" \
    "LOOK FOR: Per-tag 'wg_pop' values -- gives \\OutcomeTagPairwiseAvgFeatSel,  \\OutcomeTagPairwiseMinFeatSel,  \\OutcomeTagPairwiseMaxFeatSel" \
    "$REPO_ROOT/src/pipeline" \
    "python3 H_outcome_tag_evaluate.py --test"

# ============================================================
# Outcome tag prediction -- test run (Table 3 values)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/H_outcome_tag_evaluate.py --test --nolimits (test run)" \
    "LOOK FOR: Per-tag 'wg_pop' values -- gives \\OutcomeTagPairwiseAvg, \\OutcomeTagPairwiseMin, \\OutcomeTagPairwiseMax, \\OutcomeTagAccBaseline, \\OutcomeTagAccModel, \\OutcomeTagAccImprovement, \\DifferingOutcomePairsFrac, \\NumCuratedPredicted." \
    "$REPO_ROOT/src/pipeline" \
    "python3 H_outcome_tag_evaluate.py --test --nolimits"

# ============================================================
# SHAP importance stability (prerequisite for table printer)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/I_outcome_tag_shap_stability.py" \
    "LOOK FOR: Stability computation across 3 random splits. Writes data/outcome_tags/shap_split_stability_data.pkl (required by J_outcome_tag_results_table.py)." \
    "$REPO_ROOT/src/pipeline" \
    "python3 I_outcome_tag_shap_stability.py"

# ============================================================
# Tag results LaTeX table (Table 3) -- nolimits run
# ============================================================
run_script \
    "SCRIPT: src/pipeline/J_outcome_tag_results_table.py --nolimits" \
    "LOOK FOR: \\newcommand lines for \\OutcomeTagPairwiseAvgNolimits/MinNolimits/MaxNolimits/BSS. 'TAGS_BETTER_THAN_CHANCE_95PCT: N' gives \\NumberTagsBetterThanChance. Tag-rating Pearson correlations printed to stderr." \
    "$REPO_ROOT/src/pipeline" \
    "python3 J_outcome_tag_results_table.py --nolimits --test 2>&1"

# ============================================================
# Tag results LaTeX table -- feat-sel run (no --nolimits)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/J_outcome_tag_results_table.py --featsel" \
    "LOOK FOR: \\newcommand lines for \\OutcomeTagPairwiseAvgFeatSelJ/MinFeatSelJ/MaxFeatSelJ." \
    "$REPO_ROOT/src/pipeline" \
    "python3 J_outcome_tag_results_table.py --test 2>&1"

# ============================================================
# Extrapolated scaling: \OutcomeTagPairwiseAvgExtrapolated
# ============================================================
if [ "$QUICK" != "true" ]; then
run_script \
    "SCRIPT: src/pipeline/K_outcome_tag_extrapolate_scaling.py --test --nolimits --slowcorrectextrapolate" \
    "LOOK FOR: 'WG-POP extrapolated to N=...' values per tag -- gives \\OutcomeTagPairwiseAvgExtrapolated." \
    "$REPO_ROOT/src/pipeline" \
    "python3 K_outcome_tag_extrapolate_scaling.py --test --nolimits --slowcorrectextrapolate"
fi

# ============================================================
# Cost-effectiveness pairwise (\CostEffect) and rating-cost correlation (\RatingCostCorr)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/L_cost_effectiveness_train_and_score.py" \
    "LOOK FOR: 'ZAGG Pairwise ordering: ...' -> \\CostEffect; 'Pearson r' between rating and zagg score -> \\RatingCostCorr." \
    "$REPO_ROOT/src/pipeline" \
    "python3 L_cost_effectiveness_train_and_score.py"

# ============================================================
# Evaluation set sizes (\NOverallRatings, \NOutcomeTags, \NCostEffectiveness, \NAiForecasting)
# ============================================================
run_script \
    "SCRIPT: src/pipeline/M_data_analysis_eval_set_sizes.py" \
    "LOOK FOR: SUMMARY TABLE with total column -- gives \\NOverallRatings, \\NOutcomeTags, \\NCostEffectiveness, \\NAiForecasting." \
    "$REPO_ROOT/src/pipeline" \
    "python3 M_data_analysis_eval_set_sizes.py"

# ============================================================
# Feature lists for all 4 models
# ============================================================
run_script \
    "SCRIPT: src/pipeline/N_data_analysis_print_features.py" \
    "LOOK FOR: Feature lists for overall_rating, outcome_tag_nolimits, cost_effectiveness, and llm_forecasting models." \
    "$REPO_ROOT/src/pipeline" \
    "python3 N_data_analysis_print_features.py"

# ============================================================
# Leakage incidence and prevalence
# ============================================================
run_script \
    "SCRIPT: src/pipeline/O_leakage_report.py" \
    "LOOK FOR: Forecast-leakage (test), Forecast-leakage (val), Grade-leakage (test), Grade-leakage overall of val+test, Any leakage (test),." \
    "$REPO_ROOT/src/pipeline" \
    "python3 O_leakage_report.py"

# ============================================================
# Final notice
# ============================================================
echo "" >> "$OUT"
echo "ALL SCRIPTS COMPLETE." >> "$OUT"

echo ""
echo "All output saved to: $OUT"
echo ""
echo "Figure 2 (2-pane forecast comparison) saved to: $FIGURE2_PNG"
echo "Figure 3 (SHAP beeswarm) saved to: $SHAP_PNG"
