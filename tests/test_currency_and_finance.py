"""
Tests for data_currency_conversion.py and data_loan_disbursement.py.

Thesis §cost_effectiveness: IATI activities report planned expenditure in
various currencies and unit strings (e.g. "USD million", "£m", "Mio. EUR").
These are converted to a common USD base for cost-per-unit outcome computation.

Loan vs. disbursement classification (§baseline_methods) is required because
the thesis feature set includes whether the activity is primarily loan- or
grant-based (Bulman & Eden 2017; Goldemburg et al. 2025).
"""

from decimal import Decimal

import pytest

from data_currency_conversion import (
    _detect_currency,
    _detect_scale,
    _is_missing_unit,
    _to_float,
    convert_amount,
)
from data_loan_disbursement import (
    classify_from_finance_type,
    classify_from_misc,
    classify_from_txns,
    LOAN_FT_CODES,
    DISB_FT_CODES,
)

# ---------------------------------------------------------------------------
# _to_float
# ---------------------------------------------------------------------------


class TestToFloat:
    def test_int(self):
        assert _to_float(5) == pytest.approx(5.0)

    def test_float_passthrough(self):
        assert _to_float(3.14) == pytest.approx(3.14)

    def test_numeric_string(self):
        assert _to_float("2.5") == pytest.approx(2.5)

    def test_none_returns_none(self):
        assert _to_float(None) is None

    def test_empty_string_returns_none(self):
        assert _to_float("") is None

    def test_null_string_returns_none(self):
        assert _to_float("null") is None


# ---------------------------------------------------------------------------
# _is_missing_unit
# ---------------------------------------------------------------------------


class TestIsMissingUnit:
    @pytest.mark.parametrize("unit", ["", None])
    def test_empty_or_none(self, unit):
        assert _is_missing_unit(unit) is True

    @pytest.mark.parametrize(
        "unit",
        [
            "n/a",
            "N/A",
            "none",
            "null",
            "not available",
            "not specified",
            "not applicable",
        ],
    )
    def test_sentinel_strings(self, unit):
        assert _is_missing_unit(unit) is True

    @pytest.mark.parametrize("unit", ["USD", "EUR", "USD million", "£m", "GBP"])
    def test_valid_units_not_missing(self, unit):
        assert _is_missing_unit(unit) is False


# ---------------------------------------------------------------------------
# _detect_scale
# ---------------------------------------------------------------------------


class TestDetectScale:
    @pytest.mark.parametrize(
        "unit,expected",
        [
            ("USD million", 1e6),
            ("EUR millions", 1e6),
            ("USD mn", 1e6),
            ("GBP billion", 1e9),
            ("USD bn", 1e9),
            ("USD thousand", 1e3),
            ("USD thousands", 1e3),
            ("£m", 1e6),
            ("€m", 1e6),
            ("$m", 1e6),
            ("US$m", 1e6),
            ("Mio. EUR", 1e6),
            ("Mio EUR", 1e6),
            ("usd millions", 1e6),
            ("dollars million", 1e6),
            ("USD trillion", 1e12),
            ("USD", 1.0),  # no scale word → base unit
        ],
    )
    def test_scale_detection(self, unit, expected):
        assert _detect_scale(unit.lower()) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# _detect_currency
# ---------------------------------------------------------------------------


class TestDetectCurrency:
    @pytest.mark.parametrize(
        "unit,expected",
        [
            ("USD", "USD"),
            ("US$", "USD"),
            ("US dollars", "USD"),
            ("EUR", "EUR"),
            ("Euro", "EUR"),
            ("€", "EUR"),
            ("GBP", "GBP"),
            ("£", "GBP"),
            ("pounds sterling", "GBP"),
            ("SEK", "SEK"),
            ("PHP", "PHP"),
            ("CNY", "CNY"),
            ("FCFA", "FCFA"),
            ("CFAF", "FCFA"),
            ("XOF", "FCFA"),
            ("XAF", "FCFA"),
            ("CHF", "CHF"),
            ("SDR", "SDR"),
            ("BRL", "BRL"),
            ("USD million", "USD"),
            ("EUR million", "EUR"),
            ("£m", "GBP"),
        ],
    )
    def test_currency_detection(self, unit, expected):
        assert _detect_currency(unit) == expected

    def test_unknown_returns_none(self):
        assert _detect_currency("ZZZZ unknown stuff") is None


# ---------------------------------------------------------------------------
# convert_amount
# ---------------------------------------------------------------------------

_DUMMY_RECORD = {"activity_id": "XM-DAC-1", "section": "test"}


class TestConvertAmount:
    """Thesis §cost_effectiveness: financial amounts converted to USD for denominator."""

    def test_eur_million(self):
        # 5 EUR million → 5 * 1e6 * 1.08 = 5,400,000 USD
        result = convert_amount(
            5, "EUR million", field_name="planned", record=_DUMMY_RECORD
        )
        assert result == pytest.approx(5 * 1e6 * 1.08)

    def test_gbp_million(self):
        result = convert_amount(
            10, "GBP million", field_name="planned", record=_DUMMY_RECORD
        )
        assert result == pytest.approx(10 * 1e6 * 1.27)

    def test_usd_million_explicit(self):
        result = convert_amount(
            3, "USD million", field_name="planned", record=_DUMMY_RECORD
        )
        assert result == pytest.approx(3e6)

    def test_pound_m_shorthand(self):
        result = convert_amount(2, "£m", field_name="planned", record=_DUMMY_RECORD)
        assert result == pytest.approx(2 * 1e6 * 1.27)

    def test_none_amount_returns_none(self):
        assert (
            convert_amount(None, "USD million", field_name="x", record=_DUMMY_RECORD)
            is None
        )

    def test_zero_amount_returns_none(self):
        assert (
            convert_amount(0, "USD million", field_name="x", record=_DUMMY_RECORD)
            is None
        )

    def test_negative_amount_returns_none(self):
        assert (
            convert_amount(-5, "USD million", field_name="x", record=_DUMMY_RECORD)
            is None
        )

    def test_missing_unit_returns_none(self):
        assert convert_amount(5, "n/a", field_name="x", record=_DUMMY_RECORD) is None

    def test_unknown_currency_returns_none(self):
        assert (
            convert_amount(5, "ZZZZ million", field_name="x", record=_DUMMY_RECORD)
            is None
        )

    def test_usd_billion(self):
        result = convert_amount(1, "USD billion", field_name="x", record=_DUMMY_RECORD)
        assert result == pytest.approx(1e9)

    def test_peer_unit_infers_currency_for_bare_scale(self):
        # unit is just "million" with no currency — peer_unit provides it
        result = convert_amount(
            4,
            "million",
            field_name="x",
            record=_DUMMY_RECORD,
            peer_unit="USD million",
        )
        assert result == pytest.approx(4e6)

    def test_small_plain_usd_returns_none_ambiguous(self):
        # < $10,000 plain USD is flagged as ambiguous scale (could be "million" missing)
        assert convert_amount(100, "USD", field_name="x", record=_DUMMY_RECORD) is None

    def test_large_eur_million_rescaled_if_over_threshold(self):
        # 200,000 EUR million = 2e11 EUR > HIGH_USD_THRESHOLD after FX → rescaled /1e6
        result = convert_amount(
            200_000, "EUR million", field_name="x", record=_DUMMY_RECORD
        )
        # After rescaling: 200_000 * 1.08 (the million is already applied, then reversed)
        assert result is not None


# ---------------------------------------------------------------------------
# classify_from_finance_type  (thesis §baseline_methods: loan vs grant feature)
# ---------------------------------------------------------------------------


class TestClassifyFromFinanceType:
    @pytest.mark.parametrize(
        "code", ["410", "411", "421", "422", "431", "810", "910", "1100"]
    )
    def test_loan_codes(self, code):
        assert classify_from_finance_type(code) == "loan"

    @pytest.mark.parametrize("code", ["110", "111", "210", "211", "610", "620", "630"])
    def test_disbursement_codes(self, code):
        assert classify_from_finance_type(code) == "disbursement"

    def test_unknown_code_returns_none(self):
        assert classify_from_finance_type("999") is None

    def test_empty_code_returns_none(self):
        assert classify_from_finance_type("") is None

    def test_loan_codes_set_non_empty(self):
        assert len(LOAN_FT_CODES) > 0

    def test_disb_codes_set_non_empty(self):
        assert len(DISB_FT_CODES) > 0

    def test_loan_and_disb_sets_disjoint(self):
        assert LOAN_FT_CODES.isdisjoint(DISB_FT_CODES)


# ---------------------------------------------------------------------------
# classify_from_misc
# ---------------------------------------------------------------------------


class TestClassifyFromMisc:
    def test_loan_only(self):
        entry = {
            "loan_total": 5000,
            "loan_units": "USD",
            "disbursement_total": None,
            "disbursement_units": None,
        }
        assert classify_from_misc(entry) == "loan"

    def test_disbursement_only(self):
        entry = {
            "disbursement_total": 3000,
            "disbursement_units": "USD",
            "loan_total": None,
            "loan_units": None,
        }
        assert classify_from_misc(entry) == "disbursement"

    def test_both_none_returns_none(self):
        entry = {
            "loan_total": None,
            "disbursement_total": None,
            "loan_units": None,
            "disbursement_units": None,
        }
        assert classify_from_misc(entry) is None

    def test_both_zero_returns_none(self):
        entry = {
            "loan_total": 0,
            "disbursement_total": 0,
            "loan_units": "USD",
            "disbursement_units": "USD",
        }
        assert classify_from_misc(entry) is None

    def test_both_present_same_units_loan_larger(self):
        entry = {
            "loan_total": 8000,
            "loan_units": "USD",
            "disbursement_total": 3000,
            "disbursement_units": "USD",
        }
        assert classify_from_misc(entry) == "loan"

    def test_both_present_same_units_disb_larger(self):
        entry = {
            "loan_total": 3000,
            "loan_units": "USD",
            "disbursement_total": 8000,
            "disbursement_units": "USD",
        }
        assert classify_from_misc(entry) == "disbursement"

    def test_both_present_same_units_equal_amounts_tie_breaks_loan(self):
        entry = {
            "loan_total": 5000,
            "loan_units": "USD",
            "disbursement_total": 5000,
            "disbursement_units": "USD",
        }
        assert classify_from_misc(entry) == "loan"

    def test_both_present_different_units_returns_none(self):
        # Different units → ambiguous, cannot classify
        entry = {
            "loan_total": 5000,
            "loan_units": "USD",
            "disbursement_total": 3000,
            "disbursement_units": "EUR",
        }
        assert classify_from_misc(entry) is None


# ---------------------------------------------------------------------------
# classify_from_txns
# ---------------------------------------------------------------------------


class TestClassifyFromTxns:
    """Transaction type codes: 5/6/10 → loan; 3/4/7 → disbursement."""

    @pytest.mark.parametrize("code", ["5", "6", "10"])
    def test_loan_transaction_codes(self, code):
        txns = {code: Decimal("1000")}
        assert classify_from_txns(txns) == "loan"

    @pytest.mark.parametrize("code", ["3", "4", "7"])
    def test_disbursement_transaction_codes(self, code):
        txns = {code: Decimal("500")}
        assert classify_from_txns(txns) == "disbursement"

    def test_loan_code_takes_priority_over_disbursement(self):
        # Both loan and disbursement codes present → loan wins (checked first)
        txns = {"5": Decimal("100"), "3": Decimal("200")}
        assert classify_from_txns(txns) == "loan"

    def test_empty_txns_returns_none(self):
        assert classify_from_txns({}) is None

    def test_zero_amount_ignored(self):
        txns = {"5": Decimal("0"), "3": Decimal("500")}
        assert classify_from_txns(txns) == "disbursement"

    def test_unknown_codes_returns_none(self):
        txns = {"99": Decimal("1000")}
        assert classify_from_txns(txns) is None
