"""
Tests for rating-parsing functions in overall_rating_feature_engineering.py.

The thesis (§interpreting_ratings) defines a canonical 0-5 integer scale:
  0 = Highly Unsatisfactory … 5 = Highly Satisfactory
Activities from DE-1 organisations (BMZ/KfW/GIZ) arrive on an *inverted*
1–6 scale (1 = best) and must be flipped.  All other numeric ratings are
linearly rescaled to [0, 5].  Free-text labels are mapped via a multilingual
alias table.  FCDO/UK activities use A++/A+/A/B/C letter grades.
"""

import math

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    _norm_text,
    _strip_accents,
    _extract_numbers,
    _parse_percent,
    get_from_number,
    get_success_measure_from_rating_value,
    get_success_measure_from_rating_value_wrapped,
    parse_last_line_label_after_forecast,
    pick_start_date,
    add_enhanced_uncertainty_features,
)

# ---------------------------------------------------------------------------
# _strip_accents
# ---------------------------------------------------------------------------


class TestStripAccents:
    def test_plain_ascii_unchanged(self):
        assert _strip_accents("hello") == "hello"

    def test_e_acute_stripped(self):
        assert _strip_accents("é") == "e"

    def test_full_french_word(self):
        assert _strip_accents("très") == "tres"

    def test_german_umlaut(self):
        assert _strip_accents("ü") == "u"

    def test_empty_string(self):
        assert _strip_accents("") == ""


# ---------------------------------------------------------------------------
# _norm_text
# ---------------------------------------------------------------------------


class TestNormText:
    def test_none_returns_empty_string(self):
        assert _norm_text(None) == ""

    def test_lowercases(self):
        assert _norm_text("Satisfactory") == "satisfactory"

    def test_strips_accents(self):
        assert _norm_text("Très Satisfaisant") == "tres satisfaisant"

    def test_removes_parenthetical(self):
        # "(MS)" parenthetical should be stripped
        assert _norm_text("MS (Moderately Satisfactory)") == "ms"

    def test_strips_trailing_punctuation(self):
        assert _norm_text("satisfactory.") == "satisfactory"

    def test_collapses_whitespace(self):
        assert _norm_text("highly   satisfactory") == "highly satisfactory"

    def test_strips_leading_trailing_spaces(self):
        assert _norm_text("  good  ") == "good"


# ---------------------------------------------------------------------------
# _extract_numbers / _parse_percent
# ---------------------------------------------------------------------------


class TestExtractNumbers:
    def test_single_integer(self):
        assert _extract_numbers("3") == [3.0]

    def test_decimal(self):
        assert _extract_numbers("3.5") == [3.5]

    def test_comma_decimal(self):
        assert _extract_numbers("3,5") == [3.5]

    def test_multiple_numbers(self):
        assert _extract_numbers("score 4 out of 6") == [4.0, 6.0]

    def test_empty_string(self):
        assert _extract_numbers("") == []

    def test_none(self):
        assert _extract_numbers(None) == []


class TestParsePercent:
    def test_integer_percent(self):
        assert _parse_percent("80%") == pytest.approx(80.0)

    def test_decimal_percent(self):
        assert _parse_percent("75.5%") == pytest.approx(75.5)

    def test_no_percent_returns_none(self):
        assert _parse_percent("80") is None

    def test_none_returns_none(self):
        assert _parse_percent(None) is None


# ---------------------------------------------------------------------------
# get_success_measure_from_rating_value  (canonical text → 0-5)
# ---------------------------------------------------------------------------


class TestGetSuccessMeasureFromRatingValue:
    """Thesis §interpreting_ratings: canonical 0-5 mapping for the six labels."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("highly unsatisfactory", 0),
            ("unsatisfactory", 1),
            ("moderately unsatisfactory", 2),
            ("moderately satisfactory", 3),
            ("satisfactory", 4),
            ("highly satisfactory", 5),
        ],
    )
    def test_canonical_labels(self, label, expected):
        assert get_success_measure_from_rating_value(label) == expected

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("high", 4),
            ("medium", 2.5),
            ("low", 1),
        ],
    )
    def test_simple_three_grades(self, label, expected):
        assert get_success_measure_from_rating_value(label) == expected

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("substantial", 3),
            ("modest", 2),
            ("negligible", 1),
        ],
    )
    def test_substantial_grades(self, label, expected):
        assert get_success_measure_from_rating_value(label) == expected

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("highly successful", 5),
            ("successful", 4),
            ("unsuccessful", 1),
        ],
    )
    def test_simple_success_grades(self, label, expected):
        assert get_success_measure_from_rating_value(label) == expected

    def test_contains_highly_satisfactory(self):
        assert (
            get_success_measure_from_rating_value("overall highly satisfactory result")
            == 5
        )

    def test_contains_moderately_unsatisfactory(self):
        assert (
            get_success_measure_from_rating_value(
                "the project was moderately unsatisfactory"
            )
            == 2
        )

    def test_unknown_text_returns_none(self):
        assert get_success_measure_from_rating_value("gibberish xyz") is None

    def test_empty_string_returns_none(self):
        assert get_success_measure_from_rating_value("") is None


# ---------------------------------------------------------------------------
# get_from_number  (numeric rating + scale → 0-5)
# ---------------------------------------------------------------------------


class TestGetFromNumber:
    """Thesis §interpreting_ratings: linear rescaling of numeric ratings to [0,5].
    DE-1 (BMZ/KfW/GIZ) uses 1-6 inverted scale (1=best, 6=worst).
    """

    def test_fraction_string_format(self):
        # "3/6" → 5 * (3/6) = 2.5
        result = get_from_number("3/6", None, None, "XM-DAC-1")
        assert result == pytest.approx(2.5)

    def test_standard_numeric_rescaling(self):
        # v=3, lo=1, hi=6 → 5*(3-1)/(6-1) = 2.0
        assert get_from_number(3, 6, 1, "XM-DAC-1") == pytest.approx(2.0)

    def test_max_value_standard(self):
        # v=6, lo=1, hi=6 → 5
        assert get_from_number(6, 6, 1, "XM-DAC-1") == pytest.approx(5.0)

    def test_min_value_standard(self):
        # v=1, lo=1, hi=6 → 0
        assert get_from_number(1, 6, 1, "XM-DAC-1") == pytest.approx(0.0)

    def test_de1_inverted_best_score(self):
        # DE-1 lo=1,hi=6: v=1 (best) → 5 after inversion
        assert get_from_number(1, 6, 1, "DE-1-KfW") == pytest.approx(5.0)

    def test_de1_inverted_worst_score(self):
        # DE-1 lo=1,hi=6: v=6 (worst) → 0 after inversion
        assert get_from_number(6, 6, 1, "DE-1-KfW") == pytest.approx(0.0)

    def test_de1_mid_score(self):
        # DE-1: v=3, lo=1, hi=6 → raw=2.0, inverted=5-2.0=3.0
        assert get_from_number(3, 6, 1, "DE-1-KfW") == pytest.approx(3.0)

    def test_value_outside_range_returns_none(self):
        assert get_from_number(7, 6, 1, "XM-DAC-1") is None

    def test_hi_equals_lo_returns_none(self):
        assert get_from_number(3, 3, 3, "XM-DAC-1") is None

    def test_non_numeric_returns_none(self):
        assert get_from_number("abc", 6, 1, "XM-DAC-1") is None


# ---------------------------------------------------------------------------
# get_success_measure_from_rating_value_wrapped  (alias table + DE-1 parsing)
# ---------------------------------------------------------------------------

_WB = "XM-DAC-41114"  # World Bank — not DE-1, not GB-
_DE = "DE-1-KfW"
_GB = "GB-GOV-1-100"


class TestGetSuccessMeasureWrapped:
    """Multilingual alias table and DE-1 numeric handling (thesis §interpreting_ratings)."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("HS", 5),
            ("S", 4),
            ("MS", 3),
            ("MU", 2),
            ("U", 1),
            ("HU", 0),
        ],
    )
    def test_english_abbreviations(self, label, expected):
        result = get_success_measure_from_rating_value_wrapped(label, activity_id=_WB)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("très satisfaisant", 5),
            ("satisfaisant", 4),
            ("satisfaisant", 4),
            ("insatisfaisant", 1),
        ],
    )
    def test_french_aliases(self, label, expected):
        result = get_success_measure_from_rating_value_wrapped(label, activity_id=_WB)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("sehr gut", 5),
            ("gut", 4),
            ("erfolgreich", 4),
            ("nicht erfolgreich", 1),
        ],
    )
    def test_german_aliases(self, label, expected):
        result = get_success_measure_from_rating_value_wrapped(label, activity_id=_WB)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("altamente satisfactorio", 5),
            ("satisfactorio", 4),
            ("insatisfactorio", 1),
        ],
    )
    def test_spanish_aliases(self, label, expected):
        result = get_success_measure_from_rating_value_wrapped(label, activity_id=_WB)
        assert result == pytest.approx(expected)

    def test_percent_string(self):
        # 80% → 5 * 80/100 = 4.0
        result = get_success_measure_from_rating_value_wrapped("80%", activity_id=_WB)
        assert result == pytest.approx(4.0)

    def test_de1_numeric_string(self):
        # DE-1 with "3" out of 1-6 inverted → 3.0
        result = get_success_measure_from_rating_value_wrapped(
            "3", min_rating="1", max_rating="6", activity_id=_DE
        )
        assert result == pytest.approx(3.0)

    def test_on_track_maps_to_satisfactory(self):
        result = get_success_measure_from_rating_value_wrapped(
            "on track", activity_id=_WB
        )
        assert result == pytest.approx(4.0)

    def test_exceeded_expectations(self):
        result = get_success_measure_from_rating_value_wrapped(
            "exceeded expectations", activity_id=_WB
        )
        assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# parse_last_line_label_after_forecast
# ---------------------------------------------------------------------------

_RECORD = {"activity_id": "XM-DAC-41114"}


class TestParseLastLineLabelAfterForecast:
    """Parses the FORECAST: label from the last line of LLM text output."""

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Highly Satisfactory", 5),
            ("Satisfactory", 4),
            ("Moderately Satisfactory", 3),
            ("Moderately Unsatisfactory", 2),
            ("Unsatisfactory", 1),
            ("Highly Unsatisfactory", 0),
        ],
    )
    def test_canonical_labels(self, label, expected):
        text = f"Some reasoning text.\nFORECAST: {label}"
        assert parse_last_line_label_after_forecast(text, _RECORD) == pytest.approx(
            expected
        )

    def test_strips_markdown_bold(self):
        text = "reasoning\nFORECAST: **Satisfactory**"
        assert parse_last_line_label_after_forecast(text, _RECORD) == pytest.approx(4.0)

    def test_strips_numbered_bullet(self):
        text = "reasoning\n4. FORECAST: Moderately Satisfactory"
        assert parse_last_line_label_after_forecast(text, _RECORD) == pytest.approx(3.0)

    def test_case_insensitive_forecast_keyword(self):
        text = "reasoning\nforecast: Satisfactory"
        assert parse_last_line_label_after_forecast(text, _RECORD) == pytest.approx(4.0)

    def test_no_forecast_keyword_returns_none(self):
        text = "This response has no forecast line."
        assert parse_last_line_label_after_forecast(text, _RECORD) is None

    def test_empty_text_returns_none(self):
        assert parse_last_line_label_after_forecast("", _RECORD) is None

    def test_multiline_uses_last_line(self):
        text = "FORECAST: Highly Satisfactory\nMore text\nFORECAST: Unsatisfactory"
        assert parse_last_line_label_after_forecast(text, _RECORD) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# pick_start_date
# ---------------------------------------------------------------------------


class TestPickStartDate:
    """Uses actual_start_date → planned_start_date → txn_first_date priority."""

    def test_prefers_actual_start_date(self):
        row = pd.Series(
            {
                "actual_start_date": pd.Timestamp("2010-01-01"),
                "original_planned_start_date": pd.Timestamp("2009-06-01"),
                "txn_first_date": pd.Timestamp("2009-01-01"),
            }
        )
        assert pick_start_date(row) == pd.Timestamp("2010-01-01")

    def test_falls_back_to_planned_start_date(self):
        row = pd.Series(
            {
                "actual_start_date": pd.NaT,
                "original_planned_start_date": pd.Timestamp("2009-06-01"),
                "txn_first_date": pd.Timestamp("2009-01-01"),
            }
        )
        assert pick_start_date(row) == pd.Timestamp("2009-06-01")

    def test_falls_back_to_txn_first_date(self):
        row = pd.Series(
            {
                "actual_start_date": pd.NaT,
                "original_planned_start_date": pd.NaT,
                "txn_first_date": pd.Timestamp("2009-01-01"),
            }
        )
        assert pick_start_date(row) == pd.Timestamp("2009-01-01")

    def test_all_missing_returns_nat(self):
        row = pd.Series(
            {
                "actual_start_date": pd.NaT,
                "original_planned_start_date": pd.NaT,
                "txn_first_date": pd.NaT,
            }
        )
        assert pd.isna(pick_start_date(row))

    def test_missing_column_skipped(self):
        row = pd.Series(
            {
                "original_planned_start_date": pd.Timestamp("2011-03-15"),
            }
        )
        assert pick_start_date(row) == pd.Timestamp("2011-03-15")


# ---------------------------------------------------------------------------
# add_enhanced_uncertainty_features
# ---------------------------------------------------------------------------


class TestAddEnhancedUncertaintyFeatures:
    """Missingness indicator columns are added when their source columns exist."""

    def _base_df(self, n=20):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "finance": rng.uniform(0, 100, n),
                "complexity": rng.uniform(0, 100, n),
                "risks": rng.uniform(0, 100, n),
                "cpia_score": rng.uniform(2, 5, n),
                "gdp_percap": rng.uniform(6, 12, n),
                "planned_expenditure": rng.uniform(10, 100, n),
                "planned_duration": rng.uniform(1, 10, n),
                "umap3_x": rng.normal(0, 1, n),
                "umap3_y": rng.normal(0, 1, n),
                "umap3_z": rng.normal(0, 1, n),
            }
        )
        df.loc[df.sample(5, random_state=0).index, "cpia_score"] = np.nan
        df.loc[df.sample(3, random_state=1).index, "umap3_x"] = np.nan
        return df

    def test_llm_missing_count_added(self):
        df = self._base_df()
        add_enhanced_uncertainty_features(df)
        assert "llm_features_missing_count" in df.columns

    def test_cpia_missing_indicator_added(self):
        df = self._base_df()
        add_enhanced_uncertainty_features(df)
        assert "cpia_missing" in df.columns

    def test_cpia_missing_matches_nan(self):
        df = self._base_df()
        add_enhanced_uncertainty_features(df)
        expected = df["cpia_score"].isna().astype(float)
        pd.testing.assert_series_equal(df["cpia_missing"], expected, check_names=False)

    def test_umap_missing_indicator_added(self):
        df = self._base_df()
        add_enhanced_uncertainty_features(df)
        assert "umap_missing" in df.columns

    def test_umap_missing_is_one_where_nan(self):
        df = self._base_df()
        add_enhanced_uncertainty_features(df)
        # wherever umap3_x is NaN, umap_missing must be 1.0
        nan_rows = df["umap3_x"].isna()
        assert (df.loc[nan_rows, "umap_missing"] == 1.0).all()

    def test_gdp_percap_missing_indicator(self):
        df = self._base_df()
        df.loc[[0, 1], "gdp_percap"] = np.nan
        add_enhanced_uncertainty_features(df)
        assert "gdp_percap_missing" in df.columns
        assert df.loc[[0, 1], "gdp_percap_missing"].eq(1.0).all()

    def test_returns_dataframe(self):
        df = self._base_df()
        result = add_enhanced_uncertainty_features(df)
        assert result is df  # in-place modification
