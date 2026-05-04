# About

This codebase reproduces my thesis results reported in detail [here](https://github.com/morganrivers/forecasting_earth_system/blob/previews/master/thesis.pdf). All data used for reproducing the results are located in the `data` directory in this repository. A simpler-to-navigate, data-only repository is located [here](https://github.com/morganrivers/iati_extraction).

# How to Reproduce Thesis Results

To reproduce all thesis results in one shot, run from the repo root:

```bash
bash run_all_thesis_results.sh
```

This runs every script below in sequence, saves all terminal output to `thesis_results_output.txt`, and prints the paths to the two saved figures (Figure 2 and Figure 3) at the end. No plots will be displayed interactively.

Pass `--quick` to skip the two slow extrapolated-scaling scripts (`E_overall_rating_extrapolate_scaling.py` and `K_outcome_tag_extrapolate_scaling.py`), which produce `\PairwiseModelExtrapolated` and `\OutcomeTagPairwiseAvgExtrapolated`:

```bash
bash run_all_thesis_results.sh --quick
```

---

The tables and numbers in the thesis are produced by the scripts described below.
The latest thesis can always be found at
```https://raw.githubusercontent.com/morganrivers/forecasting_earth_system/refs/heads/main/tex/thesis.tex```

The thesis contains commands which define the key results mentioned throughout the latex file for the thesis. These are directly extracted from the `thesis_results_output.txt` file using `extract_latex_values.py`. Running that script will show you the meanings of each of the outputs. You can find the full context for each of these constants within `https://raw.githubusercontent.com/morganrivers/forecasting_earth_system/refs/heads/main/tex/thesis.tex`.

All scripts live under `src/pipeline/`. The results (three results tables, shap values for ratings, and Results constants `\newcommand` values) are populated via the following scripts:

---

## Overall rating prediction (Table 1: method comparison; R^2, RMSE, MAE, pairwise ranking)

**Script:** `src/pipeline/A_overall_rating_fit_and_evaluate.py`

Prints the full held-out test-set results table. Look for `=== RESULTS (terminal) ===` and underneath:
- `Pairwise = ...  [95% CI: ...]` -- gives `\PairwiseModel`, `\PairwiseCILow`, `\PairwiseCIHigh`
- The ridge baseline (risks + org only) pairwise line -- gives `\PairwiseHuman`
- R^2 lines per method -- gives `\Rsqfrac`, `\Rsqnollmfrac`, `\Rsqwithllmfrac` (and `\Rsqdelta` = difference)
- Adjusted R^2 on train+val for all orgs -- gives `\RsqTraining`

Also writes `data/rating_model_outputs/` (predictions, model pickle, feature importances).

```bash
cd src/pipeline
python3 A_overall_rating_fit_and_evaluate.py            # full results table
python3 A_overall_rating_fit_and_evaluate.py importances  # additionally prints feature importance table (Table 2)
```

---

## SHAP beeswarm plot (Figure 3: feature importances for rating RF)

**Script:** `src/pipeline/B_overall_rating_plot_shap.py`

Reads the saved model and training features from `data/rating_model_outputs/` (written by
`A_overall_rating_fit_and_evaluate.py`) and saves `src/pipeline/feature_importances_plot.png`.
Requires `A_overall_rating_fit_and_evaluate.py` to have been run first.

```bash
cd src/pipeline
python3 B_overall_rating_plot_shap.py
```

---

## In-sample R^2 on training+validation set (\RsqTrainingOLS, \RsqAdjTrainingOLS)

**Script:** `src/pipeline/C_overall_rating_insample_r2.py`

Prints `R^2` and `Adjusted R^2` for the OLS model fit on train+val. Requires
`data/rating_model_outputs/all_features.csv` (written by `A_overall_rating_fit_and_evaluate.py` on a prior run).

```bash
cd src/pipeline
python3 C_overall_rating_insample_r2.py
```

---

## OOS val predictions (prerequisite for narrative grading)

**Script:** `src/pipeline/D_overall_rating_generate_oos_predictions.py`

Generates `data/rf_oos_val_predictions.csv` required by `E_llm_score_forecast_narratives.py`.

```bash
cd src/pipeline
python3 D_overall_rating_generate_oos_predictions.py
```

---

## Narrative similarity grades (\BestNarrativeGrade, \RisksOnlyGrade)

**Script:** `src/pipeline/E_llm_score_forecast_narratives.py`

Calls *gemini-2.5-flash-lite* to grade each LLM forecast against the post-activity evaluation summary.
Prints `Mean Grade: <value>` per forecast config (look for the best RF-forced full-context config for
`\BestNarrativeGrade`, and the risks-only config for `\RisksOnlyGrade`). Grades cached to
`data/forecast_grades/grades_*.jsonl`.

```bash
cd src/pipeline
python3 E_llm_score_forecast_narratives.py            # displays Figure 2 interactively
python3 E_llm_score_forecast_narratives.py --no-show  # saves Figure 2 without displaying it
```

---

## Outcome tag prediction (Table 3: tag model results; \OutcomeTagPairwiseAvg etc.)

Four scripts run in sequence:

1. **`src/pipeline/F_outcome_tag_train.py --use_test --hardcode_tags`** -- fits RF+ET classifiers on train/val with hardcoded tags; a **required prerequisite** for the nolimits run below.

2. **`src/pipeline/F_outcome_tag_train.py --use_test --nolimits --hardcode_tags`** -- re-fits on train+val, evaluates on test set with no tag limits.

3. **`src/pipeline/G_outcome_tag_evaluate.py --test`** -- prints per-tag `wg_pop` values with feature selection -- gives `\OutcomeTagPairwiseAvgFeatSel`, `\OutcomeTagPairwiseMinFeatSel`, `\OutcomeTagPairwiseMaxFeatSel`.

4. **`src/pipeline/G_outcome_tag_evaluate.py --test --nolimits`** -- prints per-tag `wg_pop` values for the full model -- gives `\OutcomeTagPairwiseAvg`, `\OutcomeTagPairwiseMin`, `\OutcomeTagPairwiseMax`, `\OutcomeTagAccBaseline`, `\OutcomeTagAccModel`, `\OutcomeTagAccImprovement`, `\DifferingOutcomePairsFrac`, `\NumCuratedPredicted`.

```bash
cd src/pipeline
python3 F_outcome_tag_train.py --use_test --hardcode_tags
python3 F_outcome_tag_train.py --use_test --nolimits --hardcode_tags
python3 G_outcome_tag_evaluate.py --test
python3 G_outcome_tag_evaluate.py --test --nolimits
```

---

## SHAP importance stability (prerequisite for Table 3 printer)

**Script:** `src/pipeline/H_outcome_tag_shap_stability.py`

Computes SHAP importance stability across 3 random splits, writes
`data/outcome_tags/shap_split_stability_data.pkl` (required by `I_outcome_tag_results_table.py`).

```bash
cd src/pipeline
python3 H_outcome_tag_shap_stability.py
```

---

## Tag results LaTeX table (Table 3) and tag-rating correlations

**Script:** `src/pipeline/I_outcome_tag_results_table.py`

Prints the LaTeX table (Table 3) with consistent SHAP features and `\newcommand` lines for
`\OutcomeTagPairwiseAvg/Min/Max/BSS`. Also prints `TAGS_BETTER_THAN_CHANCE_95PCT: N` (gives
`\NumberTagsBetterThanChance`) and tag-rating Pearson correlations -- gives `\TagCorrTargetsMet` etc.
Look for `-- Tag correlation with six_overall_rating (all activities) --`.

```bash
cd src/pipeline
python3 I_outcome_tag_results_table.py --nolimits --test 2>&1
```

---

## Extrapolated scaling values (\PairwiseModelExtrapolated, \OutcomeTagPairwiseAvgExtrapolated)

- **`\PairwiseModelExtrapolated`** -- from `src/pipeline/J_data_analysis_extrapolate_scaling.py --lc-pop`
  (`analysis_17_learning_curve`), prints `pop_proj_5x` (log-linear extrapolation to 5x training data for the overall rating pairwise ranking skill of the RF+ET model).

- **`\OutcomeTagPairwiseAvgExtrapolated`** -- from `src/pipeline/K_outcome_tag_extrapolate_scaling.py --test --nolimits --slowcorrectextrapolate`,
  which fits an exponential saturation curve to each tag's learning data and extrapolates; prints
  `Within Group Pairwise Ranking Skill extrapolated to N=...` values per tag.

```bash
cd src/pipeline
python3 J_data_analysis_extrapolate_scaling.py --lc-pop

python3 K_outcome_tag_extrapolate_scaling.py --test --nolimits --slowcorrectextrapolate
```

---

## Cost-effectiveness pairwise ranking (\CostEffect) and rating-cost correlation (\RatingCostCorr)

**Script:** `src/pipeline/L_cost_effectiveness_train_and_score.py`

Prints `ZAGG Pairwise ordering: ...` (gives `\CostEffect`) and
`Pearson r` between rating and zagg score (gives `\RatingCostCorr`).

```bash
cd src/pipeline
python3 L_cost_effectiveness_train_and_score.py
```

---

## Evaluation set sizes (\NOverallRatings, \NOutcomeTags, \NCostEffectiveness, \NAiForecasting)

**Script:** `src/pipeline/M_data_analysis_eval_set_sizes.py`

Prints a SUMMARY TABLE with total column -- gives `\NOverallRatings`, `\NOutcomeTags`,
`\NCostEffectiveness`, `\NAiForecasting`.

```bash
cd src/pipeline
python3 M_data_analysis_eval_set_sizes.py
```

---

## Feature lists for all 4 models

**Script:** `src/pipeline/N_data_analysis_print_features.py`

---

## Leakage incidence and prevalence

**Script:** `src/pipeline/O_leakage_report.py`

Prints leakage counts and rates across splits from `data/LEAKAGERISK.json`.

```bash
cd src/pipeline
python3 O_leakage_report.py
```

Prints the feature lists for the overall rating, outcome tag (nolimits), cost-effectiveness, and LLM forecasting models. Feature lists are saved to `data/feature_lists/` by their respective training scripts (A, G, L); this script reads and displays them.

```bash
cd src/pipeline
python3 N_data_analysis_print_features.py
```

---

## Tag-rating correlations (\TagCorrTargetsMet, \TagCorrBeneficiarySatisfaction, etc.)

Printed by `src/pipeline/I_outcome_tag_results_table.py` (see above). Look for
`-- Tag correlation with six_overall_rating (all activities) --` in output.

---

## Fine-tuning results (\FineTuningLossReduction, \FineTuningEpochs, \FineTuningPairs, \FineTuningCostEuros)

Fine-tuning was run once via `src/pipeline/B_fine_tune_2p5_pro.py` (requires GCP credentials
and a GCS bucket). Metric values (loss reduction, epochs, pairs, cost) are hardcoded constants recorded
from that run. The fine-tuned model forecasts live in
`data/rag_prompts_and_responses/outputs_*fine_tuning*.jsonl` and can be evaluated by enabling
the fine-tuning configs in `FORECAST_CONFIGS` at the top of `E_llm_score_forecast_narratives.py`.






---

## Leakage handling (`data/LEAKAGERISK.json`)

Some activities in the val and test sets contain post-cutoff information that leaked into inputs used by the pipeline. Two kinds of leakage are tracked:

- **Grade leakage** (`*grades*` source keys, e.g. `section_texts_val_grades`): the LLM-graded feature files for these activities referenced post-cutoff events (e.g. COVID-19) in the project document sections used for grading. Affects 15 activities across val and test.
- **Forecast leakage** (`*forecast*` source keys, e.g. `deepseek_minimal_val_forecast`, `glm_rating_forecast_test`): the LLM narrative forecast for these activities directly mentioned post-cutoff events. Affects 75 activities (val + test). 12 activities have both kinds.

The top-level key `exclude_test_leakage_risk` (default `true`) enables leakage handling. Set it to `false` to run without any leakage correction as a comparison baseline.

### How leakage is handled per script

**`A_overall_rating_fit_and_evaluate.py`** — controlled by `LEAKAGE_HANDLING_METHOD` (top of file):

| Mode | Effect |
|------|--------|
| `"replace_predictions"` (default) | Sets grade features to `NaN` for grade-leakage activities before training (RF fills from training-set medians). After all predictions are built, overwrites LLM-blended prediction columns for test leakage rows: grade-leakage rows → `pred_rf_no_llm`; forecast-only rows → `pred_rf` (base RF, no LLM narrative). |
| `"median_impute"` | Only the NaN step above; no prediction swap. Useful to isolate the training effect. |
| `"drop"` | Removes all leakage activities from `test_idx` (original behaviour). |

**`G_outcome_tag_train.py`** — drops test-set grade-leakage activities from evaluation (unchanged; outcome-tag evaluation is harder to substitute).

**`F_llm_score_forecast_narratives.py`** — for each forecast config that has a `leakage_source_key` matching a LEAKAGERISK source, the affected activity IDs are removed from `valid_ids` before metrics are computed. Currently wired for `deepseek_minimal_val_forecast`.

### Adding new leakage entries

Add an entry to `data/LEAKAGERISK.json` with the structure:
```json
"<activity_id>": {
  "activity_id": "<activity_id>",
  "split": "val" | "test",
  "leakage_sources": {
    "<source_name>": ["<reason>"]
  }
}
```
Source name conventions: end with `_grades` for grade leakage, `_forecast` for narrative forecast leakage. To wire a new forecast source into F_, add `"leakage_source_key": "<source_name>"` to its entry in `FORECAST_CONFIGS`.

---

## Data

The data-only repository is hosted at https://github.com/morganrivers/iati_extraction . This repo also contains all data necessary to run the code, which is roughly a subset of the full dataset.

### Overall ratings
- `data/merged_overall_ratings.jsonl` — overall activity success ratings on a ~1–6 scale (IEG/DSGF ratings)

### Outcome tags
- `data/outcome_tags/applied_tags.jsonl` — binary outcome tags extracted by LLM; each entry has `activity_id` and a dict of `tag_name → binary value`

### Outcome summaries
- `data/outputs_summary_expost.jsonl` — LLM-generated ex-post summaries of activity outcomes (used as context when grading forecasts)

### Quantitative outcomes
- `data/activity_outcomes.csv` — quantitative outcome values (CO2, beneficiaries, costs, etc.) with `outcome_norm`, `baseline_norm`, `target_norm`, `outcome_norm_dollars_per_unit`, etc.

### Features
- `data/rating_model_outputs/all_features.csv` — all ~58 feature values for every activity (train+val+test) with split label

Features include: LLM-extracted activity properties (finance, integratedness, implementer_performance, targets, context, risks, complexity, activity_scope), country characteristics (gdp_percap, cpia_score, WGI governance indicators), region dummies, planned_duration, planned_expenditure, UMAP embeddings (umap3_x/y/z), sector_distance, country_distance, sector_cluster dummies, missingness indicators, and org dummies.

### LLM-extracted pre-activity features
LLM-graded activity properties (1–6 scale):
- `data/outputs_finance_grades.jsonl` → `finance` feature
- `data/outputs_integratedness_grades.jsonl` → `integratedness` feature
- `data/outputs_implementer_performance_grades.jsonl` → `implementer_performance` feature
- `data/outputs_targets_grades.jsonl` → `targets` feature
- `data/outputs_context_grades.jsonl` → `context` feature
- `data/outputs_risks_grades.jsonl` → `risks` feature
- `data/outputs_complexity_grades.jsonl` → `complexity` feature

Other LLM feature sources:
- `data/outputs_targets_context_maps.jsonl` — pairwise similarity maps (val set); used for `sector_distance`, `country_distance`
- `data/outputs_targets_context_maps_trainval.jsonl` — same for train+val (test set run)
- `data/outputs_misc.jsonl` — miscellaneous LLM extractions (`activity_scope` and other fields)
- `data/llm_planned_expenditure.jsonl` → `planned_expenditure` feature
- `data/llm_planned_duration.jsonl` → `planned_duration` feature
- `data/outputs_finance_sectors_disbursements_baseline_gemini2p5flash.jsonl` → `sector_cluster_*` features

### LLM forecasts
Free-form forecast texts, in `data/rag_prompts_and_responses/`. Files with `val` in the name are the ~300-activity validation set (used in the 2-pane comparison figure); `no_knn_no_rag_deepseek_minimal_val` also covers ~199 test activities.

Key validation-set files:
- `outputs_exactly_like_halawi_et_al_rag_added_gemini3pro_val_s3_call_1.jsonl` — gemini-2.5-pro (KNN+RAG+S1+S2)
- `outputs_exactly_like_halawi_et_al_rag_added_deepseek_minimal_val_s3_call_1.jsonl` — deepseek-v3.2 (KNN+RAG+S1+S2)
- `outputs_exactly_like_halawi_et_al_rag_added_forced_rf_deepseek_val_forced_rf_s3_call_1.jsonl` — deepseek-v3.2 RF forced (KNN+RAG+S1+S2)
- `outputs_exactly_like_halawi_et_al_rag_added_no_knn_no_rag_forced_rf_deepseek_val_no_knn_no_rag_forced_rf_s3_call_1.jsonl` — deepseek-v3.2 RF forced (no KNN, no RAG)
- `outputs_exactly_like_halawi_et_al_rag_added_deepseek_minimal_val_no_goodbadcalls_s3_call_1.jsonl` — deepseek-v3.2 (KNN+RAG, no S1/S2)
- `outputs_exactly_like_halawi_et_al_rag_added_no_knn_no_rag_deepseek_minimal_val_s3_call_1.jsonl` — deepseek-v3.2 (no KNN, no RAG, no S1, no S2); also serves as test set predictions
- `outputs_onlysummary_no_knn_no_rag_onlysummary_no_knn_no_rag_s3_call_1.jsonl` — deepseek-v3.2 (minimal prompt, summary only)

### LLM narrative forecast grades
Raw grades in `data/forecast_grades/grades_*.jsonl` — output of the grading LLM per activity, including `response_text` (reasoning + `GRADE:` letter), `activity_id`, and token usage.
