"""
Extracts thesis result values from thesis_results_output.txt and prints
\newcommand lines. Prints % NOT FOUND when a value cannot be located.

Every value is extracted from its specific SCRIPT: section so that
duplicate patterns across sections (e.g. --nolimits vs non-nolimits
H_outcome_tag_evaluate runs) never bleed into the wrong variable.
"""

import re
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
TXT = REPO_ROOT / "thesis_results_output.txt"

if not TXT.exists():
    print(f"ERROR: {TXT} not found", file=sys.stderr)
    sys.exit(1)

raw = TXT.read_text(errors="replace")
# Strip ANSI escape codes so numeric values aren't split by color sequences
text = re.sub(r"\x1b\[[0-9;]*m", "", raw)


def pct(v):
    """Float 0-1 -> integer percent (round half-up)."""
    return int(math.floor(v * 100 + 0.5))


def emit(name, value, comment=None):
    if comment:
        print(f"% {comment}")
    if value is None:
        print(f"% NOT FOUND: \\{name}")
        print(f"\\newcommand{{\\{name}}}{{NOT FOUND}}")
    else:
        print(f"\\newcommand{{\\{name}}}{{{value}}}")


def get_section(heading_pattern):
    """Return the body text of the section whose SCRIPT: heading matches
    heading_pattern, up to (but not including) the next SCRIPT: heading."""
    m = re.search(heading_pattern, text)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"\nSCRIPT:", text[start:])
    end = start + nxt.start() if nxt else len(text)
    return text[start:end]


def wg_pop_from_section(section_text):
    """Extract wg_pop values from the WG-POP DEBUG table inside a section.
    Returns list of floats for rf+ET rows, or None if not found."""
    m = re.search(
        r"\[WG-POP DEBUG\].*?wg_pop\b\s*\n\s*-+\s*\n(.*?)-{3,}", section_text, re.DOTALL
    )
    if not m:
        return None
    pops = re.findall(r"rf\+ET\s+([0-9.]+)", m.group(1))
    return [float(x) for x in pops] if pops else None


# ---------------------------------------------------------------------------
# Build all sections up-front so every extraction is explicitly scoped
# ---------------------------------------------------------------------------
evalset_sec = get_section(r"SCRIPT: src/pipeline/M_data_analysis_eval_set_sizes\.py")
glm_sec = get_section(r"SCRIPT: src/pipeline/A_overall_rating_fit_and_evaluate\.py")
variance_sec = get_section(r"SCRIPT: src/pipeline/C_overall_rating_insample_r2\.py")
narrative_sec = get_section(r"SCRIPT: src/pipeline/F_llm_score_forecast_narratives\.py")
featsel_sec = get_section(
    r"SCRIPT: src/pipeline/H_outcome_tag_evaluate\.py --test \(test run\)"
)
nolimits_sec = get_section(
    r"SCRIPT: src/pipeline/H_outcome_tag_evaluate\.py --test --nolimits \(test run\)"
)
print_tag_sec = get_section(r"SCRIPT: src/pipeline/J_outcome_tag_results_table\.py --nolimits")
print_tag_featsel_sec = get_section(r"SCRIPT: src/pipeline/J_outcome_tag_results_table\.py --featsel")
learning_curve_sec = get_section(
    r"SCRIPT: src/pipeline/E_overall_rating_extrapolate_scaling\.py"
)
extrap_sec = get_section(r"SCRIPT: src/pipeline/K_outcome_tag_extrapolate_scaling\.py")
zagg_sec = get_section(r"SCRIPT: src/pipeline/L_cost_effectiveness_train_and_score\.py")


# ---------------------------------------------------------------------------
# \PairwiseHuman -- WG Pair. Rank. of "Ridge Baseline (risks + org only)"
# Section: A_overall_rating_fit_and_evaluate.py
# ---------------------------------------------------------------------------
m = re.search(
    r"Ridge Baseline \(risks \+ org only\)\s+R2=[^\n]+WG Pair\. Rank\.=(0\.\d+)",
    glm_sec,
)
emit(
    "PairwiseHuman",
    pct(float(m.group(1))) if m else None,
    "Pairwise ranking accuracy (%) of the Ridge Baseline (risks + org features only, no LLM) on the held-out test set",
)

# ---------------------------------------------------------------------------
# \PairwiseModel, \PairwiseCILow, \PairwiseCIHigh
# Section: A_overall_rating_fit_and_evaluate.py  (BOOTSTRAP 95% CI block)
# ---------------------------------------------------------------------------
m = re.search(
    r"BOOTSTRAP 95% CI: RF\+ET.*?"
    r"Pairwise\s*=\s*(0\.\d+)\s+\[95% CI:\s*(0\.\d+),\s*(0\.\d+)\]",
    glm_sec,
    re.DOTALL,
)
if m:
    emit(
        "PairwiseModel",
        pct(float(m.group(1))),
        "Pairwise ranking accuracy (%) of the RF+ET ensemble (all features + year corr) on the held-out test set",
    )
    emit(
        "PairwiseCILow",
        pct(float(m.group(2))),
        "Lower bound of bootstrap 95% CI for PairwiseModel (%)",
    )
    emit(
        "PairwiseCIHigh",
        pct(float(m.group(3))),
        "Upper bound of bootstrap 95% CI for PairwiseModel (%)",
    )
else:
    emit(
        "PairwiseModel",
        None,
        "Pairwise ranking accuracy (%) of the RF+ET ensemble (all features + year corr) on the held-out test set",
    )
    emit("PairwiseCILow", None, "Lower bound of bootstrap 95% CI for PairwiseModel (%)")
    emit(
        "PairwiseCIHigh", None, "Upper bound of bootstrap 95% CI for PairwiseModel (%)"
    )

# ---------------------------------------------------------------------------
# \PairwiseModelExtrapolated
# Section: E_overall_rating_extrapolate_scaling.py
# ---------------------------------------------------------------------------
m = re.search(r"pop_proj_5x\D+(0\.\d+)", learning_curve_sec)
emit(
    "PairwiseModelExtrapolated",
    pct(float(m.group(1))) if m else None,
    "Projected pairwise ranking accuracy (%) of RF+ET extrapolated to 5x the current training population size",
)

print()

# ---------------------------------------------------------------------------
# \Rsqfrac -- AccInt of "RF+ET all features + year corr" from results table
# Table columns: N  R2  RMSE  MAE  SideAcc  AccInt  Spearman  WGSpearman  WGPairRank
# Section: A_overall_rating_fit_and_evaluate.py
# ---------------------------------------------------------------------------
m = re.search(
    r"RF\+ET all features \+ year corr\s+"
    r"\d+\s+"  # N
    r"[-0-9.]+\s+"  # R2
    r"[0-9.]+\s+"  # RMSE
    r"[0-9.]+\s+"  # MAE
    r"[0-9.]+\s+"  # SideAcc
    r"([0-9.]+)",  # AccInt <- Rsqfrac
    glm_sec,
)
emit(
    "Rsqfrac",
    m.group(1) if m else None,
    "AccInt of RF+ET (all features + year corr) on held-out test set: fraction of predictions within the correct integer rating interval (raw float, e.g. 0.52)",
)

# ---------------------------------------------------------------------------
# \Rsqheldoutratingsllmfeatures -- R2 of "RF+ET all features + year corr"
# Section: A_overall_rating_fit_and_evaluate.py
# ---------------------------------------------------------------------------
m = re.search(r"RF\+ET all features \+ year corr\s+R2=(-?[0-9.]+)", glm_sec)
emit(
    "Rsqheldoutratingsllmfeatures",
    m.group(1) if m else None,
    "R^2 of RF+ET (all features including LLM features, with year correction) on the held-out test set",
)

# ---------------------------------------------------------------------------
# \Rsqheldoutridgebaseline -- R2 of "Ridge Baseline (risks + org only)"
# Section: A_overall_rating_fit_and_evaluate.py
# ---------------------------------------------------------------------------
m = re.search(r"Ridge Baseline \(risks \+ org only\)\s+R2=(-?[0-9.]+)", glm_sec)
emit(
    "Rsqheldoutridgebaseline",
    m.group(1) if m else None,
    "R^2 of the Ridge Baseline (risks + org features only, no LLM features) on the held-out test set",
)

# ---------------------------------------------------------------------------
# \Rsqnollmfrac -- R2 of "RF+ET, no LLM features"
# Section: A_overall_rating_fit_and_evaluate.py
# ---------------------------------------------------------------------------
m = re.search(r"RF\+ET,\s*no LLM features\s+R2=(-?[0-9.]+)", glm_sec)
emit(
    "Rsqnollmfrac",
    m.group(1) if m else None,
    "R^2 of RF+ET trained without any LLM-derived features on the held-out test set",
)

# ---------------------------------------------------------------------------
# \Rsqwithllmfrac -- R2 of "RF+ET all features (no year corr)"
# Section: A_overall_rating_fit_and_evaluate.py
# ---------------------------------------------------------------------------
m = re.search(r"RF\+ET all features \(no year corr\)\s+R2=(-?[0-9.]+)", glm_sec)
emit(
    "Rsqwithllmfrac",
    m.group(1) if m else None,
    "R^2 of RF+ET with all features but without year correction on the held-out test set",
)

# ---------------------------------------------------------------------------
# \CostEffect, \CostCILow, \CostCIHigh
# Section: L_cost_effectiveness_train_and_score.py
# ---------------------------------------------------------------------------
m = re.search(
    r"ZAGG Pairwise ordering:\s*(0\.\d+)\s+95% CI:\s*\[(0\.\d+),\s*(0\.\d+)\]", zagg_sec
)
if m:
    emit(
        "CostEffect",
        pct(float(m.group(1))),
        "Pairwise ordering accuracy (%) of RF+ET for predicting cost-effectiveness (ZAGG score) on the held-out test set",
    )
    emit(
        "CostCILow", pct(float(m.group(2))), "Lower bound of 95% CI for CostEffect (%)"
    )
    emit(
        "CostCIHigh", pct(float(m.group(3))), "Upper bound of 95% CI for CostEffect (%)"
    )
else:
    emit(
        "CostEffect",
        None,
        "Pairwise ordering accuracy (%) of RF+ET for predicting cost-effectiveness (ZAGG score) on the held-out test set",
    )
    emit("CostCILow", None, "Lower bound of 95% CI for CostEffect (%)")
    emit("CostCIHigh", None, "Upper bound of 95% CI for CostEffect (%)")

print()

# ---------------------------------------------------------------------------
# \OutcomeTagPairwiseAvg/Min/Max -- averaged across nolimits and feat-sel J runs
# Sections: J_outcome_tag_results_table.py --nolimits  and  --featsel
# ---------------------------------------------------------------------------
def _avg_j_metric(base_cmd, desc_avg, desc_nl):
    m_nl = re.search(rf"\\newcommand{{\\{base_cmd}Nolimits}}{{(\d+)}}", print_tag_sec)
    m_fs = re.search(rf"\\newcommand{{\\{base_cmd}FeatSelJ}}{{(\d+)}}", print_tag_featsel_sec)
    if not m_nl:
        print(f"ERROR: \\{base_cmd}Nolimits not found in J --nolimits section", file=sys.stderr)
        sys.exit(1)
    if not m_fs:
        print(f"ERROR: \\{base_cmd}FeatSelJ not found in J --featsel section", file=sys.stderr)
        sys.exit(1)
    nl_val = int(m_nl.group(1))
    fs_val = int(m_fs.group(1))
    val = int(round((nl_val + fs_val) / 2))
    emit(base_cmd, val, desc_avg)
    emit(f"{base_cmd}Nolimits", nl_val, desc_nl)

_avg_j_metric(
    "OutcomeTagPairwiseAvg",
    "Average WG pairwise ranking accuracy (%) across all outcome tag classifiers on the test set (averaged over nolimits and feat-sel runs)",
    "Average WG pairwise ranking accuracy (%) across all outcome tag classifiers on the test set (nolimits run only)",
)
_avg_j_metric(
    "OutcomeTagPairwiseMin",
    "Lowest WG pairwise ranking accuracy (%) among all outcome tag classifiers on the test set (averaged over nolimits and feat-sel runs)",
    "Lowest WG pairwise ranking accuracy (%) among all outcome tag classifiers on the test set (nolimits run only)",
)
_avg_j_metric(
    "OutcomeTagPairwiseMax",
    "Highest WG pairwise ranking accuracy (%) among all outcome tag classifiers on the test set (averaged over nolimits and feat-sel runs)",
    "Highest WG pairwise ranking accuracy (%) among all outcome tag classifiers on the test set (nolimits run only)",
)

# ---------------------------------------------------------------------------
# \OutcomeTagPairwiseAvgFeatSel/MinFeatSel/MaxFeatSel
# Section: H_outcome_tag_evaluate.py --test  (feat-sel, no --nolimits)
# ---------------------------------------------------------------------------
vals_fs = wg_pop_from_section(featsel_sec)
if vals_fs:
    emit(
        "OutcomeTagPairwiseAvgFeatSel",
        pct(sum(vals_fs) / len(vals_fs)),
        "Average WG pairwise ranking accuracy (%) across outcome tag classifiers after feature selection on the test set",
    )
    emit(
        "OutcomeTagPairwiseMinFeatSel",
        pct(min(vals_fs)),
        "Lowest WG pairwise ranking accuracy (%) among outcome tag classifiers after feature selection on the test set",
    )
    emit(
        "OutcomeTagPairwiseMaxFeatSel",
        pct(max(vals_fs)),
        "Highest WG pairwise ranking accuracy (%) among outcome tag classifiers after feature selection on the test set",
    )
else:
    emit(
        "OutcomeTagPairwiseAvgFeatSel",
        None,
        "Average WG pairwise ranking accuracy (%) across outcome tag classifiers after feature selection on the test set",
    )
    emit(
        "OutcomeTagPairwiseMinFeatSel",
        None,
        "Lowest WG pairwise ranking accuracy (%) among outcome tag classifiers after feature selection on the test set",
    )
    emit(
        "OutcomeTagPairwiseMaxFeatSel",
        None,
        "Highest WG pairwise ranking accuracy (%) among outcome tag classifiers after feature selection on the test set",
    )

# ---------------------------------------------------------------------------
# \OutcomeTagPairwiseAvgExtrapolated
# Section: K_outcome_tag_extrapolate_scaling.py --test --nolimits --slowcorrectextrapolate
# ---------------------------------------------------------------------------
extrap_m = re.search(
    r"EXTRAPOLATED PERFORMANCE.*?WG-POP @ N=5000\s*\n\s*-+\s*\n(.*?)(?:\n\s*\n)",
    extrap_sec,
    re.DOTALL,
)
if extrap_m:
    pops = re.findall(r"(0\.\d+)", extrap_m.group(1))
    emit(
        "OutcomeTagPairwiseAvgExtrapolated",
        pct(sum(float(x) for x in pops) / len(pops)) if pops else None,
        "Average extrapolated WG pairwise ranking accuracy (%) across outcome tag classifiers projected to N=5000 activities",
    )
else:
    emit(
        "OutcomeTagPairwiseAvgExtrapolated",
        None,
        "Average extrapolated WG pairwise ranking accuracy (%) across outcome tag classifiers projected to N=5000 activities",
    )

# ---------------------------------------------------------------------------
# \OutcomeTagBSS -- \newcommand line from J_outcome_tag_results_table.py
# ---------------------------------------------------------------------------
m = re.search(r"\\newcommand{\\OutcomeTagBSS}{([0-9.]+)}", print_tag_sec)
emit(
    "OutcomeTagBSS",
    m.group(1) if m else None,
    "Brier Skill Score of outcome tag predictions relative to baseline (higher is better; 0 = no skill)",
)

# ---------------------------------------------------------------------------
# \NumberTagsBetterThanChance -- count of tags with p < 0.05 (permutation test)
# Section: J_outcome_tag_results_table.py
# ---------------------------------------------------------------------------
m = re.search(r"TAGS_BETTER_THAN_CHANCE_95PCT:\s*(\d+)", print_tag_sec)
emit(
    "NumberTagsBetterThanChance",
    m.group(1) if m else None,
    "Number of outcome tags (out of 14 curated) with WG-POP p < 0.05 under permutation test",
)

# ---------------------------------------------------------------------------
# \OutcomeTagAccBaseline, \OutcomeTagAccModel, \OutcomeTagAccImprovement
# Section: H_outcome_tag_evaluate.py --test --nolimits
#
# H prints TWO weighted-accuracy blocks with identical line formats:
#
#   Block A  "WEIGHTED ACCURACY -- ALL N real-model tags (excl. const_base fallbacks)"
#            Covers ALL tags that have a trained model (curated + non-curated),
#            excludes tags that fell back to const_base.
#
#   Block B  "WEIGHTED ACCURACY -- curated N tags (...)"
#            Covers only the 14 curated TAG_GROUPS tags; INCLUDES const_base
#            fallbacks so the denominator is always the full curated set.
#
# We anchor each regex to its block header so there is no ambiguity.
# Currently emitting from Block B (curated tags) because Table 3 and all other
# tag reporting in the thesis is scoped to the curated 14.
# ---------------------------------------------------------------------------
_all_tags_baseline_m = re.search(
    r"WEIGHTED ACCURACY -- ALL \d+ real-model tags.*?"
    r"Constant baseline \(majority class\):\s*([0-9.]+)%",
    nolimits_sec,
    re.DOTALL,
)
_all_tags_model_m = re.search(
    r"WEIGHTED ACCURACY -- ALL \d+ real-model tags.*?"
    r"Chosen model\s*:\s*([0-9.]+)%",
    nolimits_sec,
    re.DOTALL,
)
_curated_baseline_m = re.search(
    r"WEIGHTED ACCURACY -- curated \d+ tags.*?"
    r"Constant baseline \(majority class\):\s*([0-9.]+)%",
    nolimits_sec,
    re.DOTALL,
)
_curated_model_m = re.search(
    r"WEIGHTED ACCURACY -- curated \d+ tags.*?"
    r"Chosen model\s*:\s*([0-9.]+)%",
    nolimits_sec,
    re.DOTALL,
)

# Emit from Block B (curated tags). Change the source variables below to
# _all_tags_baseline_m / _all_tags_model_m to switch to Block A instead.
baseline_m = _curated_baseline_m
model_m = _curated_model_m

emit(
    "OutcomeTagAccBaseline",
    baseline_m.group(1) if baseline_m else None,
    "Weighted accuracy (%) of the constant majority-class baseline for the 14 curated outcome tags (Block B: curated tags incl. const_base fallbacks)",
)
emit(
    "OutcomeTagAccModel",
    model_m.group(1) if model_m else None,
    "Weighted accuracy (%) of the chosen model for the 14 curated outcome tags (Block B: curated tags incl. const_base fallbacks)",
)
if baseline_m and model_m:
    improvement = round(float(model_m.group(1)) - float(baseline_m.group(1)), 1)
    emit(
        "OutcomeTagAccImprovement",
        improvement,
        "Percentage point improvement in weighted accuracy over the majority-class baseline for the 14 curated outcome tags",
    )
else:
    emit(
        "OutcomeTagAccImprovement",
        None,
        "Percentage point improvement in weighted accuracy over the majority-class baseline for the 14 curated outcome tags",
    )

# ---------------------------------------------------------------------------
# \NumCuratedPredicted -- "N real-model tags" count
# Section: H_outcome_tag_evaluate.py --test --nolimits
# ---------------------------------------------------------------------------
m = re.search(r"(\d+)\s+real.model tags", nolimits_sec)
emit(
    "NumCuratedPredicted",
    m.group(1) if m else None,
    "Number of activities in the test set that received at least one outcome tag prediction from the real (non-curated) model",
)

# ---------------------------------------------------------------------------
# \DifferingOutcomePairsFrac
# Section: H_outcome_tag_evaluate.py --test --nolimits
# Printed as: "DifferingOutcomePairsFrac ... X.X%  (n/N pairs)"
# ---------------------------------------------------------------------------
m = re.search(r"DifferingOutcomePairsFrac.*?:\s*([0-9.]+)%", nolimits_sec)
emit(
    "DifferingOutcomePairsFrac",
    int(float(m.group(1)) + 0.5) if m else None,
    "Percentage of activity pairs in the test set that have at least one differing outcome tag (denominator for pairwise ranking evaluation)",
)

print()
print("% Narrative similarity grades")

# ---------------------------------------------------------------------------
# \BestNarrativeGrade -- Mean Grade from specific deepseek-v3.2 RF forced config
# Section: F_llm_score_forecast_narratives.py
# ---------------------------------------------------------------------------
m = re.search(
    r"--- Metrics: deepseek-v3\.2 RF forced \(KNN\+RAG\+S1\+S2\)[^\n]*\n"
    r"(?:(?!--- Metrics:).)*?Mean Grade:\s*([0-9.]+)",
    narrative_sec,
    re.DOTALL,
)
if m:
    emit(
        "BestNarrativeGrade",
        f"{float(m.group(1)):.2f}",
        "Mean narrative similarity grade (0-1 scale) of the best-performing model: deepseek-v3.2 RF forced (KNN+RAG+S1+S2)",
    )
else:
    grades = re.findall(r"Mean Grade:\s*([0-9.]+)", narrative_sec)
    emit(
        "BestNarrativeGrade",
        f"{max(float(g) for g in grades):.2f}" if grades else None,
        "Mean narrative similarity grade (0-1 scale) of the best-performing model across all configurations",
    )

# ---------------------------------------------------------------------------
# \RisksOnlyGrade -- Human proxy mean grade (risks summaries)
# Section: F_llm_score_forecast_narratives.py
# ---------------------------------------------------------------------------
m = re.search(r"Human proxy mean grade:\s*([0-9.]+)", narrative_sec)
emit(
    "RisksOnlyGrade",
    f"{float(m.group(1)):.2f}" if m else None,
    "Mean narrative similarity grade (0-1 scale) when using risks-only summaries as a human-proxy baseline",
)

print()
print("% Rating vs cost-effectiveness correlation")

# ---------------------------------------------------------------------------
# \RatingCostCorr -- Pearson r (entire dataset, rating vs zagg score)
# Section: L_cost_effectiveness_train_and_score.py
# ---------------------------------------------------------------------------
m = re.search(r"PEARSON CORRELATION FOR ENTIRE DATASET\s+r\s*=\s*(-?[0-9.]+)", zagg_sec)
if not m:
    m = re.search(r"Pearson r=([0-9.]+)", zagg_sec)
emit(
    "RatingCostCorr",
    m.group(1) if m else None,
    "Pearson r between overall activity ratings and cost-effectiveness (ZAGG) scores across the entire dataset",
)

print()
print("% Within-training-set R^2")

# ---------------------------------------------------------------------------
# \RsqTraining -- R^2 from RF+ET on train+val set (REDUCED ADJUSTED R^2 block)
# Section: A_overall_rating_fit_and_evaluate.py
# ---------------------------------------------------------------------------
m = re.search(
    r"REDUCED \(ADJUSTED\) R.*?TRAIN\+VAL SET.*?"
    r"Overall\s+: n=\s*\d+\s+R\^2=([0-9.]+)",
    glm_sec,
    re.DOTALL,
)
emit(
    "RsqTraining",
    f"{float(m.group(1)):.2f}" if m else None,
    "Reduced (adjusted) R^2 of RF+ET on the combined train+val set (in-sample fit, adjusted for number of features)",
)

# ---------------------------------------------------------------------------
# \RsqTrainingOLS, \RsqAdjTrainingOLS
# Section: C_overall_rating_insample_r2.py
# ---------------------------------------------------------------------------
m = re.search(
    r"R\^2\s+=\s+([0-9.]+)\s*\n\s*Adjusted R\^2\s+=\s+([0-9.]+)", variance_sec
)
if m:
    emit(
        "RsqTrainingOLS",
        f"{float(m.group(1)):.2f}",
        "OLS R^2 on the training set (unadjusted, measures in-sample fit of the linear model)",
    )
    emit(
        "RsqAdjTrainingOLS",
        f"{float(m.group(2)):.2f}",
        "Adjusted OLS R^2 on the training set (penalises for number of predictors)",
    )
else:
    emit(
        "RsqTrainingOLS",
        None,
        "OLS R^2 on the training set (unadjusted, measures in-sample fit of the linear model)",
    )
    emit(
        "RsqAdjTrainingOLS",
        None,
        "Adjusted OLS R^2 on the training set (penalises for number of predictors)",
    )

print()
print("% Tag--rating Pearson correlations")

# ---------------------------------------------------------------------------
# Tag correlations -- from J_outcome_tag_results_table.py
# Section: J_outcome_tag_results_table.py
# Block header: -- Tag correlation with six_overall_rating (all activities) --
# ---------------------------------------------------------------------------
TAG_MAP = {
    "TagCorrTargetsMet": "targets_met_or_exceeded_success",
    "TagCorrBeneficiarySatisfaction": "high_beneficiary_satisfaction_or_reach_success",
    "TagCorrPrivateSectorEngagement": "private_sector_engagement_success",
    "TagCorrPolicyReforms": "policy_regulatory_reforms_success_success",
    "TagCorrCapacityBuilding": "capacity_building_delivered_success",
    "TagCorrFinancialPerformance": "improved_financial_performance",
    "TagCorrActivitiesNotCompleted": "funds_cancelled_or_unutilized",
    "TagCorrImplementationDelays": "project_restructured",
    "TagCorrProjectRestructured": "funds_reallocated",
    "TagCorrTargetsRevised": "targets_revised",
}

TAG_DESCRIPTIONS = {
    "TagCorrTargetsMet": "Pearson r between 'targets met or exceeded' outcome tag and six_overall_rating (all activities)",
    "TagCorrBeneficiarySatisfaction": "Pearson r between 'high beneficiary satisfaction or reach' outcome tag and six_overall_rating (all activities)",
    "TagCorrPrivateSectorEngagement": "Pearson r between 'private sector engagement success' outcome tag and six_overall_rating (all activities)",
    "TagCorrPolicyReforms": "Pearson r between 'policy/regulatory reforms success' outcome tag and six_overall_rating (all activities)",
    "TagCorrCapacityBuilding": "Pearson r between 'capacity building delivered' outcome tag and six_overall_rating (all activities)",
    "TagCorrFinancialPerformance": "Pearson r between 'improved financial performance' outcome tag and six_overall_rating (all activities)",
    "TagCorrActivitiesNotCompleted": "Pearson r between 'funds cancelled or unutilized' outcome tag and six_overall_rating (all activities)",
    "TagCorrImplementationDelays": "Pearson r between 'project restructured' outcome tag and six_overall_rating (all activities)",
    "TagCorrProjectRestructured": "Pearson r between 'funds reallocated' outcome tag and six_overall_rating (all activities)",
    "TagCorrTargetsRevised": "Pearson r between 'targets revised' outcome tag and six_overall_rating (all activities)",
}

tag_corr = {}
for m in re.finditer(
    r"Tag correlation with six_overall_rating \(all activities\)[^\n]*\n"
    r"(.*?)(?:\u2500{2,}|\n\n)",
    print_tag_sec,
    re.DOTALL,
):
    entries = re.findall(r"\s+([\w_]+)\s+r=([+\-][0-9.]+)", m.group(1))
    if entries:
        tag_corr = dict(entries)
        break

for cmd, tag in TAG_MAP.items():
    emit(cmd, tag_corr.get(tag), TAG_DESCRIPTIONS[cmd])

print()
print("% Pairs with differing outcomes (test set)")

# ---------------------------------------------------------------------------
# \DifferingOutcomePairsFrac already emitted above (after OutcomeTagAccModel)
# Repeated here as a reminder comment only -- do not emit twice
# ---------------------------------------------------------------------------

print()
print("% Evaluation set sizes (total N per dataset)")

# ---------------------------------------------------------------------------
# \NOverallRatings, \NOutcomeTags, \NCostEffectiveness, \NAiForecasting
# Section: M_data_analysis_eval_set_sizes.py  (SUMMARY TABLE, total column)
# ---------------------------------------------------------------------------
for cmd, pattern, desc in [
    (
        "NOverallRatings",
        r"overall_ratings\s+\d+\s+\d+\s+\d+\s+(\d+)",
        "Total number of activities in the overall ratings evaluation dataset (train + val + test)",
    ),
    (
        "NOutcomeTags",
        r"outcome_tags\s+\d+\s+\d+\s+\d+\s+(\d+)",
        "Total number of activities in the outcome tags evaluation dataset (train + val + test)",
    ),
    (
        "NCostEffectiveness",
        r"cost_effectiveness\s+\d+\s+\d+\s+\d+\s+(\d+)",
        "Total number of activities in the cost-effectiveness evaluation dataset (train + val + test)",
    ),
    (
        "NAiForecasting",
        r"ai_forecasting\s+\S+\s+\d+\s+\S+\s+(\d+)",
        "Total number of activities in the AI forecasting evaluation dataset",
    ),
]:
    m = re.search(pattern, evalset_sec)
    emit(cmd, m.group(1) if m else None, desc)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------


def extract_tables(section_text):
    """Return list of (label, table_text) for every \\begin{table}...\\end{table} block."""
    results = []
    for m in re.finditer(
        r"(\\begin\{table[^}]*\}.*?\\end\{table\*?\})", section_text, re.DOTALL
    ):
        block = m.group(1)
        lm = re.search(r"\\label\{([^}]+)\}", block)
        label = lm.group(1) if lm else None
        results.append((label, block))
    return results


def emit_table(label, table_text):
    tag = f"\\label{{{label}}}" if label else "(no label)"
    divider = "=" * 60
    print(f"\n{divider}")
    print(f"TABLE: {tag}")
    print(divider)
    print(table_text.strip())


print("\n\n% ================================================================")
print("% LATEX TABLES")
print("% ================================================================")

leakage_sec = get_section(r"SCRIPT: src/pipeline/O_leakage_report\.py")

print("\n\n% ================================================================")
print("% LEAKAGE REPORT (O_leakage_report.py)")
print("% ================================================================")
if leakage_sec.strip():
    print(leakage_sec.strip())
else:
    print("% NOT FOUND: O_leakage_report.py output")

features_sec = get_section(r"SCRIPT: src/pipeline/N_data_analysis_print_features\.py")
if features_sec.strip():
    print("\n\n% ================================================================")
    print("% FEATURE LISTS (N_data_analysis_print_features.py)")
    print("% ================================================================")
    print(features_sec.strip())

# Feature importance table (Top 25 Features) -- from A_overall_rating_fit_and_evaluate.py
tables_glm = extract_tables(glm_sec)
for label, tbl in tables_glm:
    # assign known labels to unlabelled tables by content
    if label is None and r"\Delta R^2" in tbl and "Feature" in tbl:
        label = "tab:feature_importance"
    elif label is None and r"WG Pair. Rank." in tbl:
        label = "tab:method_comparison_validation"
    emit_table(label, tbl)

# Tag results table -- from J_outcome_tag_results_table.py
for label, tbl in extract_tables(print_tag_sec):
    if label == "tab:tag_model_results_sig":
        continue
    emit_table(label, tbl)

# Eval set sizes table -- from M_data_analysis_eval_set_sizes.py
for label, tbl in extract_tables(evalset_sec):
    emit_table(label, tbl)
