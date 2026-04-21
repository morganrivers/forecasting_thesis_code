"""
Currency conversion utilities for IATI financial data.

Converts monetary amounts (reported in various currencies and unit strings) to
log(USD) using approximate FX rates. Handles common IATI formats including
scale words ("million", "thousand", "£m"), currency symbols (€, £, $), ISO-3
codes, and multilingual currency aliases.

Sentinels (zero, empty string, None, -1) are treated as missing and return None.
Extreme values outside [LOW_THRESH, UPPER_THRESH] are rescaled by +/-1e6 if
RESCALE_EXTREMES_BY_MILLION is True.

Main export: return_misc_disbursement_or_planned_disbursement(FORECAST_CSV, OUT_MISC)
"""

import json
import math
import re
from pathlib import Path

import pandas as pd

import csv
import numpy as np

PATH = Path("../../data/outputs_misc.jsonl")
BINS = 30
USE_LOG10 = False
VERBOSE = False
# If True, allows "$" with no other hints to be treated as USD.
ASSUME_BARE_DOLLAR_IS_USD = True

# If True, *guesses* that plain "USD"/"US$" + small decimal magnitude is "million".
# I leave this False by default (safer; matches your earlier preference to not guess).
ASSUME_SMALL_DECIMALS_ARE_MILLIONS = False

# Heuristic: rescale extreme values by 1e6
RESCALE_EXTREMES_BY_MILLION = True
LOW_USD_THRESHOLD = 150  # if < $150, treat as "missing million" => * 1e6
HIGH_USD_THRESHOLD = (
    75_000_000_000  # if > $75B, treat as "over-applied million" => / 1e6
)

LOW_LOG = 5.0
UPPER_LOG = 25
LOW_THRESH = float(np.exp(LOW_LOG))  # ~148.413
UPPER_THRESH = float(np.exp(UPPER_LOG))  # ~148.413


# Approx FX rates -> USD (edit as needed)
FX = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "SEK": 0.095,
    "PHP": 0.018,
    "CNY": 0.14,
    "TND": 0.32,
    "FCFA": 0.0017,
    "MXN": 0.059,  # approx
    "PEN": 0.27,  # approx
    "BRL": 0.20,  # approx
    "CHF": 1.26,  # approx
    "SDR": 1.44,
    "CAD": 0.74,
    # IFI units (unsupported unless you provide rates)
    "UC": None,
    "UA": None,
}

MISSING_UNIT_PATTERNS = [
    "n/a",
    "not available",
    "none",
    "null",
    "not specified",
    "units not specified",
    "unit not specified",
    "no specific",
    "no disbursements",
    "no loans",
    "no loans or credit",
    "not applicable",
    "not mentioned",
    "identified",
]

# --- add near top ---

CURRENCY_ALIASES = {
    # Common "dollar" strings in your data
    "DOLLAR": "USD",
    "DOLLARS": "USD",
    "US DOLLAR": "USD",
    "US DOLLARS": "USD",
    "US$": "USD",
    # CFA franc variants seen in your logs
    "CFAF": "FCFA",  # keep your existing FX key name
    "XOF": "FCFA",
    "XAF": "FCFA",
    # Other names you logged
    "CEDIS": "GHS",
    "NEPALI RUPEES": "NPR",
    "YUAN RENMINBI": "CNY",
}

ISO3_RE = re.compile(r"^[A-Z]{3}$")


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"null", "none"}:
            return None
        return float(s)
    return None


def _norm_unit(u):
    if u is None:
        return ""
    if not isinstance(u, str):
        u = str(u)
    u = u.replace("\n", " ").strip()
    u = re.sub(r"\s+", " ", u)
    return u


def _is_missing_unit(unit: str) -> bool:
    if not unit:
        return True
    ul = unit.lower().strip()
    for p in MISSING_UNIT_PATTERNS:
        if p in ul:
            return True
    return False


def _detect_scale(u_lower: str) -> float:
    # base is 1 currency unit
    scale = 1.0
    # German million abbreviations: "Mio.", "Mio"
    if re.search(r"\bmio\.?\b", u_lower):
        scale = max(scale, 1e6)

    # "trillion" (you logged "trillion Som")
    if re.search(r"\btrillion\b", u_lower):
        scale = max(scale, 1e12)

    # thousands
    if re.search(r"\b(thousand|thousands)\b", u_lower) or re.search(
        r"us\$\s*thousands", u_lower
    ):
        scale = max(scale, 1e3)
    if re.search(r"\b(k)\b", u_lower) and "sek" not in u_lower:
        scale = max(scale, 1e3)

    # million / billion
    if re.search(r"\b(billion|bn)\b", u_lower):
        scale = max(scale, 1e9)
    if re.search(r"\b(million|mn|millions)\b", u_lower):
        scale = max(scale, 1e6)

    # compact forms: £m, €m, $m, us$m, ( $m. ), etc.
    if re.search(r"(£\s*m\b|£m\b)", u_lower):
        scale = max(scale, 1e6)
    if re.search(r"(€\s*m\b|€m\b)", u_lower):
        scale = max(scale, 1e6)
    if re.search(r"(\$\s*m\b|\$m\b)", u_lower):
        scale = max(scale, 1e6)
    if re.search(r"(us\$\s*m\b|us\$m\b|us\$\s*m\.\b|us\$m\.\b)", u_lower):
        scale = max(scale, 1e6)

    # odd phrasing: "us dollars million"
    if (
        "dollars million" in u_lower
        or "usd millions" in u_lower
        or "usd million" in u_lower
    ):
        scale = max(scale, 1e6)

    return scale


def _detect_currency(u: str) -> str | None:
    u = _norm_unit(u)
    ul = u.lower().strip()

    # 0) If the unit is basically a code (or an alias), normalize it.
    #    This catches: "SDR", "INR", "CHF", "BDT", "CFAF", "USD", etc.
    codeish = re.sub(r"\s+", " ", u).strip().upper()
    if ISO3_RE.match(codeish) or codeish in CURRENCY_ALIASES:
        code = CURRENCY_ALIASES.get(codeish, codeish)
        return code

    if re.search(r"\buc\b", ul) or "uc (units of account)" in ul:
        return "UC"
    if re.search(r"\bua\b", ul) or "unit of account" in ul:
        return "UA"

    if "€" in u or "eur" in ul or "euro" in ul:
        return "EUR"
    if (
        "£" in u
        or "gbp" in ul
        or "pounds sterling" in ul
        or "pound sterling" in ul
        or "british pounds" in ul
        or "livres sterling" in ul
    ):
        return "GBP"

    usd_markers = [
        "usd",
        "us$",
        "u.s. dollar",
        "u.s. dollars",
        "us dollar",
        "us dollars",
        "united states dollar",
        "united states dollars",
        "usdollars",
        "u$s",
        "us $",
        "$usd",
        "dollars",  # <- add this if (and only if) your dataset is effectively USD here
    ]
    if any(m in ul for m in usd_markers):
        return "USD"
    if "$" in u and ASSUME_BARE_DOLLAR_IS_USD:
        return "USD"

    # Add a couple you logged as free-text, in case they appear outside ISO3:
    if "inr" in ul or "rupee" in ul and "nepali" not in ul:
        return "INR"
    if "bdt" in ul or "taka" in ul:
        return "BDT"
    if "chf" in ul or "swiss franc" in ul:
        return "CHF"
    if "npr" in ul or "nepali rupees" in ul:
        return "NPR"
    if "cedi" in ul or "ghs" in ul:
        return "GHS"

    # your existing others...
    if "php" in ul or "philippine" in ul or "php" in ul.replace(".", ""):
        return "PHP"
    if "cny" in ul or "rmb" in ul:
        return "CNY"
    if "sek" in ul:
        return "SEK"
    if "tnd" in ul:
        return "TND"
    if "fcfa" in ul or "cfaf" in ul or "xof" in ul or "xaf" in ul:
        return "FCFA"
    if "mxn" in ul:
        return "MXN"
    if re.search(r"\bpen\b", ul):
        return "PEN"
    if "r$" in ul or "brl" in ul:
        return "BRL"

    return None


def _infer_currency_from_peer(peer_unit: str) -> str | None:
    if not peer_unit:
        return None
    peer = _norm_unit(peer_unit)
    if _is_missing_unit(peer):
        return None
    return _detect_currency(peer)


def convert_amount(amount_raw, unit_raw, *, field_name, record, peer_unit=None):
    amt = _to_float(amount_raw)
    unit = _norm_unit(unit_raw)

    # Missing/sentinel amounts
    if amt is None or amt == 0 or amt == -1 or amt == -0.0:
        return None

    # Missing/sentinel units
    if _is_missing_unit(unit):
        return None

    # Negative other than sentinel
    if amt < 0:
        if VERBOSE:
            print(f"FAIL negative amount for {field_name}: {record}")
        return None

    u_lower = unit.lower()

    scale = _detect_scale(u_lower)
    currency = _detect_currency(unit)

    # If unit is just "million"/"thousand" with no currency, infer from peer field
    if currency is None:
        if re.fullmatch(
            r"(million|millions|mn|billion|bn|thousand|thousands|k|\(\$m\.\)|\(\$m\))",
            u_lower.strip(),
        ):
            currency = _infer_currency_from_peer(peer_unit or "")
    # Another common case: unit contains only scale and punctuation e.g. "($m.)"
    if currency is None and any(
        tok in u_lower
        for tok in ["million", "thousand", "($m", "$ million", "million $", "million £"]
    ):
        currency = _infer_currency_from_peer(peer_unit or "")

    if currency is None:
        if VERBOSE:
            print(f"NO_MATCH unknown currency unit='{unit}' for {field_name}: {record}")
        return None

    fx = FX.get(currency, None)
    if fx is None:
        if VERBOSE:
            print(
                f"NO_MATCH unsupported currency '{currency}' unit='{unit}' for {field_name}: {record}"
            )
        return None

    # Ambiguity: plain USD/US$ with small magnitude could be "million" missing
    looks_plain = (
        scale == 1.0
        and currency == "USD"
        and not re.search(
            r"\b(million|mn|millions|billion|bn|thousand|thousands|k)\b", u_lower
        )
        and not re.search(r"(us\$m|\$m|£m|€m)", u_lower)
    )
    if looks_plain and abs(amt) < 10000:
        if ASSUME_SMALL_DECIMALS_ARE_MILLIONS and (
            isinstance(amount_raw, float)
            or (isinstance(amount_raw, str) and "." in amount_raw)
        ):
            scale = 1e6
        else:
            if VERBOSE:
                print(
                    f"FAIL ambiguous scale (plain USD + small magnitude) for {field_name}: {record}"
                )
            return None

    usd_value = amt * scale * fx

    if RESCALE_EXTREMES_BY_MILLION:
        # Only do this when the unit didn't already clearly specify thousand/million/billion
        has_explicit_scale = bool(
            re.search(
                r"\b(thousand|thousands|k|million|millions|mn|billion|bn)\b", u_lower
            )
        ) or bool(re.search(r"(us\$m|\$m|£m|€m)", u_lower))

        if not has_explicit_scale:
            # very small -> likely missing "million"
            if usd_value < LOW_USD_THRESHOLD:
                usd_value *= 1e6

        # very large -> likely you multiplied by 1e6 but it was already base units somewhere upstream
        # (this one only really makes sense when you *did* apply a million-scale)
        if scale >= 1e6 and usd_value > HIGH_USD_THRESHOLD:
            usd_value /= 1e6

    return usd_value


def get_loans_disbursements():
    loans = []
    disb = []
    n_lines = 0

    with PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1

            try:
                outer = json.loads(line)
            except Exception:
                if VERBOSE:
                    print(f"FAIL bad outer json line {n_lines}: {line[:200]}")
                continue

            activity_id = outer.get("activity_id")
            response_text = outer.get("response_text", "{}")

            try:
                inner = json.loads(response_text)
            except Exception:
                if VERBOSE:
                    print(
                        f"FAIL bad response_text json activity_id={activity_id}: {outer}"
                    )
                continue

            record = {
                "activity_id": activity_id,
                "section": outer.get("section"),
                "loan_total": inner.get("loan_total"),
                "loan_units": inner.get("loan_units"),
                "disbursement_total": inner.get("disbursement_total"),
                "disbursement_units": inner.get("disbursement_units"),
            }

            loan_usd = convert_amount(
                inner.get("loan_total"),
                inner.get("loan_units"),
                field_name="loan_total",
                record=record,
                peer_unit=inner.get("disbursement_units"),
            )
            if loan_usd is not None:
                loans.append(loan_usd)

            disb_usd = convert_amount(
                inner.get("disbursement_total"),
                inner.get("disbursement_units"),
                field_name="disbursement_total",
                record=record,
                peer_unit=inner.get("loan_units"),
            )
            if disb_usd is not None:
                disb.append(disb_usd)

    print(f"Read {n_lines} lines")
    print(f"Converted loans: {len(loans)}")
    print(f"Converted disbursements: {len(disb)}")

    def prep(vals):
        if not USE_LOG10:
            return [v / 1e6 for v in vals], "USD (millions)"
        return [math.log10(v) for v in vals if v > 0], "log10(USD)"

    loans_x, loans_label = prep(loans)
    disb_x, disb_label = prep(disb)

    return loans_x, loans_label, disb_x, disb_label


def _as_float_money(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    # allow commas just in case
    s = s.replace(",", "")
    try:
        v = float(s)
    except Exception:
        return None
    if v <= 0:
        return None
    return v


def load_forecast_meta(path: Path):
    """
    Returns:
      scope_by_aid: dict[str, str]
      planned_by_aid: dict[str, float]  # original_planned_expenditure
      actual_by_aid: dict[str, float]   # actual_total_expenditure
    """
    scope_by_aid: dict[str, str] = {}
    planned_by_aid: dict[str, float] = {}
    actual_by_aid: dict[str, float] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            aid = (r.get("activity_id") or "").strip()
            if not aid:
                continue

            scope = (r.get("activity_scope") or "").strip()
            scope_by_aid[aid] = scope

            planned = _as_float_money(r.get("original_planned_expenditure"))
            if planned is not None:
                planned_by_aid[aid] = planned

            actual = _as_float_money(r.get("actual_total_expenditure"))
            if actual is not None:
                actual_by_aid[aid] = actual

    return scope_by_aid, planned_by_aid, actual_by_aid


def return_misc_disbursement_or_planned_disbursement(FORECAST_CSV, OUT_MISC):
    _, planned_by_aid, _ = load_forecast_meta(Path(FORECAST_CSV))
    aids = set(planned_by_aid)

    def pick_amount(aid):
        if planned_by_aid.get(aid) is not None and planned_by_aid.get(aid) != 0:
            return np.log(planned_by_aid.get(aid))
        return None

    df = pd.DataFrame(
        {
            "activity_id": list(aids),
            "planned_expenditure": [pick_amount(aid) for aid in aids],
        }
    ).set_index("activity_id")

    return df
