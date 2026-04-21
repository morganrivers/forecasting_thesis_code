"""
Tests for src/utils/overall_rating_feature_labels.py

Covers:
  - every explicitly mapped name returns its documented label
  - fallback replaces underscores with spaces
  - SHORT_NAMES is non-empty and all values are non-empty strings
  - no duplicate values (each label is unique)
"""

import pytest

import overall_rating_feature_labels as gnfl
from overall_rating_feature_labels import get_display_name, SHORT_NAMES

# -- SHORT_NAMES dict ----------------------------------------------------------


class TestShortNamesDict:
    def test_nonempty(self):
        assert len(SHORT_NAMES) > 0

    def test_all_keys_are_strings(self):
        for k in SHORT_NAMES:
            assert isinstance(k, str), f"Key {k!r} is not a string"

    def test_all_values_are_nonempty_strings(self):
        for k, v in SHORT_NAMES.items():
            assert (
                isinstance(v, str) and len(v) > 0
            ), f"Value for {k!r} is empty or not a string"

    def test_no_duplicate_labels(self):
        values = list(SHORT_NAMES.values())
        assert len(values) == len(set(values)), "Duplicate labels found in SHORT_NAMES"

    def test_no_key_equals_its_value(self):
        """Keys should be code names, values should be human-readable."""
        for k, v in SHORT_NAMES.items():
            assert k != v, f"Key equals value for {k!r}"


# -- Known mappings ------------------------------------------------------------


class TestKnownMappings:
    @pytest.mark.parametrize(
        "feature,expected",
        [
            ("rep_org_0", "Reporting org: FCDO (UK)"),
            ("rep_org_1", "Reporting org: Asian Dev. Bank"),
            ("rep_org_2", "Reporting org: World Bank"),
            ("rep_org_3", "Reporting org: BMZ"),
            ("umap3_x", "Document embedding X axis"),
            ("umap3_y", "Document embedding Y axis"),
            ("umap3_z", "Document embedding Z axis"),
            ("context", "External context (LLM: 100=enabling, 0=hostile)"),
            ("targets", "Target achievability (LLM: 100=easy, 0=impossible)"),
            ("risks", "Risk outlook (LLM: 100=low risk, 0=high risk)"),
            (
                "implementer_performance",
                "Implementer quality (LLM: 100=excellent, 0=poor)",
            ),
            ("finance", "Finance adequacy (LLM: 100=well-funded, 0=underfunded)"),
            ("complexity", "Implementation ease (LLM: 100=simple, 0=complex)"),
            ("integratedness", "Programme integration (LLM: 100=cohesive, 0=one-off)"),
            ("cpia_score", "Country governance quality (CPIA)"),
            ("gdp_percap", "GDP per capita"),
            ("planned_duration", "Planned duration"),
            ("planned_expenditure", "Planned expenditure"),
            ("log_planned_expenditure", "Log planned expenditure"),
            ("finance_is_loan", "Finance type: loan"),
            ("cpia_missing", "Governance data missing"),
            ("region_AFE", "Region: Africa (East)"),
            ("region_AFW", "Region: Africa (West)"),
            ("region_EAP", "Region: East Asia & Pacific"),
            ("region_ECA", "Region: Europe & Central Asia"),
            ("region_LAC", "Region: Latin America & Caribbean"),
            ("region_MENA", "Region: Middle East & North Africa"),
            ("region_SAS", "Region: South Asia"),
            ("country_distance", "Country dissimilarity"),
            ("sector_distance", "Sector dissimilarity"),
        ],
    )
    def test_known_mapping(self, feature, expected):
        assert get_display_name(feature) == expected


# -- Fallback behaviour --------------------------------------------------------


class TestFallback:
    def test_unknown_feature_replaces_underscores(self):
        assert get_display_name("some_unknown_feature") == "some unknown feature"

    def test_unknown_feature_no_underscores(self):
        assert get_display_name("plainname") == "plainname"

    def test_empty_string(self):
        assert get_display_name("") == ""

    def test_multiple_underscores(self):
        result = get_display_name("a_b_c_d")
        assert result == "a b c d"

    def test_leading_trailing_underscore(self):
        result = get_display_name("_hidden_")
        assert result == " hidden "

    def test_returns_string(self):
        assert isinstance(get_display_name("anything"), str)

    @pytest.mark.parametrize(
        "feature",
        [
            "my_feature",
            "foo_bar_baz",
            "x",
            "unknown_col_123",
        ],
    )
    def test_fallback_always_returns_string(self, feature):
        result = get_display_name(feature)
        assert isinstance(result, str)
        assert len(result) == len(
            feature
        )  # same length, underscores replaced by spaces
