"""
Single source of truth for human-readable feature display names used in plots
and LaTeX tables.

Usage:
    from overall_rating_feature_labels import get_display_name, SHORT_NAMES

    label = get_display_name("rep_org_0")          # -> "Reporting org: FCDO (UK)"
    label = get_display_name("unknown_col")        # -> "unknown col"  (fallback)
"""

# ---------------------------------------------------------------------------
# LLM grading prompt definitions
# Source: src/C_forecast_outcomes/A_grade_baseline_features_gpt3p5.py
#
# context (line 233):
#   "100 is a very good context conducive to activity success, 0 is a very
#    challenging context." -> HIGH = enabling external conditions, LOW = hostile.
#
# risks (line 189):
#   "100 being very little risk (and thus, likely to be very successful),
#    0 being extremely high risk." -> HIGH = low risk / safe, LOW = high risk.
#    INVERTED from what "risk level" implies -- positive correlation with rating
#    is expected and correct.
#
# targets (line 150):
#   "100 being extremely easily achieved and 0 being nearly impossible."
#    -> HIGH = easy targets, LOW = near-impossible targets.
#    NOT "ambition" -- it is achievability / ease.
#
# implementer_performance (line 105):
#   "0 (extremely bad) and 100 (extremely good) for the likely performance
#    of the organization(s)." -> HIGH = strong implementing org.
#
# finance (line 283):
#   "a+ is very well financed ... f is a very challenging financial situation"
#    -> converted to numeric; HIGH = well-funded relative to scope.
#
# complexity (line 338):
#   "100 is very simple to implement and 0 is very complex."
#    -> HIGH = simple project, LOW = complex. INVERTED from "complexity level".
#
# integratedness (line 387):
#   "100 is a very cohesive, large, well-integrated program,
#    0 is a very independent, one-off small program."
#    -> HIGH = well-embedded in a wider programme.
# ---------------------------------------------------------------------------

SHORT_NAMES: dict[str, str] = {
    "rep_org_0": "Reporting org: FCDO (UK)",
    "rep_org_1": "Reporting org: Asian Dev. Bank",
    "rep_org_2": "Reporting org: World Bank",
    "rep_org_3": "Reporting org: BMZ",
    "umap3_x": "Document embedding X axis",
    "umap3_y": "Document embedding Y axis",
    "umap3_z": "Document embedding Z axis",
    # LLM grades -- see prompt definitions above for correct interpretation
    "context": "External context (LLM: 100=enabling, 0=hostile)",
    "targets": "Target achievability (LLM: 100=easy, 0=impossible)",
    "risks": "Risk outlook (LLM: 100=low risk, 0=high risk)",
    "implementer_performance": "Implementer quality (LLM: 100=excellent, 0=poor)",
    "finance": "Finance adequacy (LLM: 100=well-funded, 0=underfunded)",
    "complexity": "Implementation ease (LLM: 100=simple, 0=complex)",
    "integratedness": "Programme integration (LLM: 100=cohesive, 0=one-off)",
    "activity_scope": "Geographic scope",
    "cpia_score": "Country governance quality (CPIA)",
    "gdp_percap": "GDP per capita",
    "governance_composite": "Governance composite index",
    "planned_duration": "Planned duration",
    "planned_expenditure": "Planned expenditure",
    "log_planned_expenditure": "Log planned expenditure",
    "expenditure_per_year_log": "Annual expenditure (log)",
    "expenditure_x_complexity": "Expenditure x complexity",
    "finance_is_loan": "Finance type: loan",
    "cpia_missing": "Governance data missing",
    "governance_missing_count": "Governance missing count",
    "region_AFE": "Region: Africa (East)",
    "region_AFW": "Region: Africa (West)",
    "region_EAP": "Region: East Asia & Pacific",
    "region_ECA": "Region: Europe & Central Asia",
    "region_LAC": "Region: Latin America & Caribbean",
    "region_MENA": "Region: Middle East & North Africa",
    "region_SAS": "Region: South Asia",
    "country_distance": "Country dissimilarity",
    "sector_distance": "Sector dissimilarity",
    "sector_cluster_Capacity_Building_and_Technical_Assistance": "Sector: capacity building",
    "sector_cluster_Contingencies": "Sector: contingencies",
    "sector_cluster_Project_Management": "Sector: project management",
}


def get_display_name(feature: str) -> str:
    """Return a human-readable label for *feature*, falling back to
    replacing underscores with spaces if no explicit mapping exists."""
    return SHORT_NAMES.get(feature, feature.replace("_", " "))
