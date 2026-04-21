"""
Builds the feature matrix for overall rating prediction from raw IATI activity
data, including organisation, sector, financial, scope, and similarity features.
"""

from collections import Counter
import re
import unicodedata
import pprint
from typing import Dict, Any, Iterable, List, Optional, Set
from pathlib import Path
import json
import glob

from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
from data_sector_clusters import process_finance_sectors_to_clusters

ACTIVITY_SCOPES = {
    "global": 7,
    "regional": 6,
    "multi-national": 5,
    "national": 4,
    "sub-national: multi-first-level administrative areas": 3,
    "sub-national: single first-level administrative area": 2,
    "sub-national: single second-level administrative area": 1,
    "single location": 0,
}

VERBOSE = False

# ---------------------------------------------------------------------------
# RATINGS
# ---------------------------------------------------------------------------
#
# CANONICAL RATING SCALE: 0-5  (six categories, integer-valued)
#
#   0 = Highly Unsatisfactory
#   1 = Unsatisfactory
#   2 = Moderately Unsatisfactory
#   3 = Moderately Satisfactory
#   4 = Satisfactory
#   5 = Highly Satisfactory
#
# This is the single source of truth for the numeric rating scale used
# throughout the entire pipeline.  All model training targets, all model
# predictions, and all clipping bounds must use [0, 5].
#
# IMPORTANT -- the raw IATI/donor data may arrive on a different scale
# (e.g. DE-1 organisations use a 1-6 integer scale where 1 = best).
# get_success_measure_from_rating_value / get_from_number normalise
# everything to [0, 5] on load.  Nothing downstream should re-introduce
# a 1-6 range.  If you see a clip to [1, 6] that is a bug.
#
# The thesis (methods) also refers to this as a 0-5 scale.  Any text
# saying "1-6 scale" in comments or strings is a legacy artefact and
# should be corrected.
# ---------------------------------------------------------------------------

RATING_MAP = {
    "Highly Unsatisfactory": 0,
    "Unsatisfactory": 1,
    "Moderately Unsatisfactory": 2,
    "Moderately Satisfactory": 3,
    "Satisfactory": 4,
    "Highly Satisfactory": 5,
}
RATING_MAP_LOWER = {k.lower(): v for k, v in RATING_MAP.items()}

# INVALID_TOKENS = {"na", "n/a", "nan", "none", "null", "", "-", "--"}

RATING_MAP_INVERSE = {v: k for k, v in RATING_MAP.items()}


def get_success_measure_from_rating_value(
    rating_value, min_rating=None, max_rating=None, activity_id=None
):
    RATING_MAP_LOCAL = {
        "highly unsatisfactory": 0,
        "unsatisfactory": 1,
        "moderately unsatisfactory": 2,
        "moderately satisfactory": 3,
        "satisfactory": 4,
        "highly satisfactory": 5,
    }
    # RATING_MAP_GRADES = {
    #     'a++': 5,
    #     'a+': 5,
    #     'a': 5/13*12,
    #     'a-': 5/13*11,
    #     'b+': 5/13*10,
    #     'b': 5/13*9,
    #     'b-': 5/13*8,
    #     'c+': 5/13*7,
    #     'c': 5/13*6,
    #     'c-': 5/13*5,
    #     'd+': 5/13*4,
    #     'd': 5/13*3,
    #     'd-': 5/13*2,
    #     'e+': 5/13*1,
    #     'e': 0,
    #     'e-': 0,
    #     'f+': 0,
    #     'f': 0,
    #     'f-': 0,
    # }

    SIMPLE_THREE_GRADES = {
        "high": 4,
        "medium": 2.5,
        "low": 1,
    }

    SUBSTANTIAL_GRADES = {
        "high": 4,
        "substantial": 3,
        "modest": 2,
        "negligible": 1,
    }

    SIMPLE_SUCCESS = {
        "highly successful": 5,
        "successful": 4,
        "unsuccessful": 1,
    }

    SIMPLE_EXPECTATIONS = {
        "exceeded expectations": 5,
        "met expectations": 3.5,
        "did not meet expectations": 1,
    }

    v = rating_value.lower()

    numeric_rating = RATING_MAP_LOCAL.get(v)
    if numeric_rating is not None:
        return numeric_rating

    simple_three = SIMPLE_THREE_GRADES.get(v)
    if simple_three is not None:
        return simple_three

    substantial_grades = SUBSTANTIAL_GRADES.get(v)
    if substantial_grades is not None:
        return substantial_grades

    simple_success = SIMPLE_SUCCESS.get(v)
    if simple_success is not None:
        return simple_success

    simple_expectations = SIMPLE_EXPECTATIONS.get(v)
    if simple_expectations is not None:
        return simple_expectations

    if "moderately unsatisfactory" in v:
        return 2
    elif "moderately satisfactory" in v:
        return 3
    elif "highly unsatisfactory" in v:
        return 0
    elif "highly satisfactory" in v:
        return 5
    elif "unsatisfactory" in v:
        return 1
    elif "satisfactory" in v:
        return 4

    if "excellent performance" in v:
        return 5
    if "poor performance" in v:
        return 1
    if "low performance" in v:
        return 1

    # if str(activity_id).startswith("DE-1"):
    # print(f"error: de-1 was not processed! Got {rating_value}, min  {min_rating}, max {max_rating}")

    return None


# ---------------------------------------------------------------------------
# EXTRA MAPPINGS
# ---------------------------------------------------------------------------


_CANON_LABEL_TO_0_5 = {
    "highly unsatisfactory": 0,
    "unsatisfactory": 1,
    "moderately unsatisfactory": 2,
    "moderately satisfactory": 3,
    "satisfactory": 4,
    "highly satisfactory": 5,
}

# Rating scale invariants -- any code that clips predictions must use [0, 5]
assert _CANON_LABEL_TO_0_5["highly unsatisfactory"] == 0, "worst rating must map to 0"
assert _CANON_LABEL_TO_0_5["highly satisfactory"] == 5, "best rating must map to 5"
assert list(_CANON_LABEL_TO_0_5.values()) == list(
    range(6)
), "canonical labels must map to consecutive integers 0-5"


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _norm_text(x) -> str:
    s = "" if x is None else str(x)
    s = _strip_accents(s).lower().strip()
    s = s.replace("-", "-").replace("--", "-")
    s = re.sub(r"\s+", " ", s)
    # drop parenthetical abbreviations like "(ms)" "(satisfactory)"
    s = re.sub(r"\([^)]*\)", "", s).strip()
    # normalize common punctuation
    s = s.strip(" .,:;*_\"'`")
    return s


_ALIAS_TO_CANON = {
    # -------------------------
    # abbreviations / shorthands
    # -------------------------
    "hs": "highly satisfactory",
    "s": "satisfactory",
    "ms": "moderately satisfactory",
    "mu": "moderately unsatisfactory",
    "u": "unsatisfactory",
    "hu": "highly unsatisfactory",
    # common alt abbreviations
    "highly sat": "highly satisfactory",
    "highly satisf.": "highly satisfactory",
    "mod sat": "moderately satisfactory",
    "mod satisf.": "moderately satisfactory",
    "mod unsat": "moderately unsatisfactory",
    "mod unsatisf.": "moderately unsatisfactory",
    "unsat": "unsatisfactory",
    "unsatisf.": "unsatisfactory",
    # -------------------------
    # French (after accent stripping)
    # -------------------------
    "tres satisfaisant": "highly satisfactory",
    "tres satisfaisante": "highly satisfactory",
    "tres insatisfaisant": "highly unsatisfactory",
    "tres insatisfaisante": "highly unsatisfactory",
    "satisfaisant": "satisfactory",
    "satisfaisante": "satisfactory",
    "moyennement satisfaisant": "moderately satisfactory",
    "moyennement satisfaisante": "moderately satisfactory",
    "partiellement satisfaisant": "moderately satisfactory",
    "partiellement satisfaisante": "moderately satisfactory",
    "insatisfaisant": "unsatisfactory",
    "insatisfaisante": "unsatisfactory",
    "plutot insatisfaisant": "moderately unsatisfactory",
    "plutot insatisfaisante": "moderately unsatisfactory",
    "excellent": "highly satisfactory",
    "tres bon": "highly satisfactory",
    "tres bonne": "highly satisfactory",
    "bon": "satisfactory",
    "bonne": "satisfactory",
    "assez bon": "moderately satisfactory",
    "assez bonne": "moderately satisfactory",
    "moyen": "moderately satisfactory",
    "moyenne": "moderately satisfactory",
    "faible": "unsatisfactory",
    "tres faible": "highly unsatisfactory",
    "reussi": "satisfactory",
    "reussie": "satisfactory",
    "non reussi": "unsatisfactory",
    "non reussie": "unsatisfactory",
    "atteint": "satisfactory",
    "atteinte": "satisfactory",
    "non atteint": "unsatisfactory",
    "non atteinte": "unsatisfactory",
    # -------------------------
    # Spanish (after accent stripping)
    # -------------------------
    "altamente satisfactoria": "highly satisfactory",
    "altamente satisfactorio": "highly satisfactory",
    "muy satisfactorio": "highly satisfactory",
    "muy satisfactoria": "highly satisfactory",
    "satisfactorio": "satisfactory",
    "satisfactoria": "satisfactory",
    "moderadamente satisfactorio": "moderately satisfactory",
    "moderadamente satisfactoria": "moderately satisfactory",
    "parcialmente satisfactorio": "moderately satisfactory",
    "parcialmente satisfactoria": "moderately satisfactory",
    "insatisfactorio": "unsatisfactory",
    "insatisfactoria": "unsatisfactory",
    "moderadamente insatisfactorio": "moderately unsatisfactory",
    "moderadamente insatisfactoria": "moderately unsatisfactory",
    "altamente insatisfactorio": "highly unsatisfactory",
    "altamente insatisfactoria": "highly unsatisfactory",
    "insuficiente": "unsatisfactory",
    "deficiente": "unsatisfactory",
    "muy deficiente": "highly unsatisfactory",
    "excelente": "highly satisfactory",
    "muy bueno": "highly satisfactory",
    "muy buena": "highly satisfactory",
    "bueno": "satisfactory",
    "buena": "satisfactory",
    "regular": "moderately satisfactory",
    "aceptable": "moderately satisfactory",
    "malo": "unsatisfactory",
    "mala": "unsatisfactory",
    "muy malo": "highly unsatisfactory",
    "muy mala": "highly unsatisfactory",
    "logrado": "satisfactory",
    "lograda": "satisfactory",
    "no logrado": "unsatisfactory",
    "no lograda": "unsatisfactory",
    "alcanzado": "satisfactory",
    "alcanzada": "satisfactory",
    "no alcanzado": "unsatisfactory",
    "no alcanzada": "unsatisfactory",
    # -------------------------
    # Portuguese (after accent stripping)
    # -------------------------
    "muito satisfatorio": "highly satisfactory",
    "muito satisfatoria": "highly satisfactory",
    "altamente satisfatorio": "highly satisfactory",
    "altamente satisfatoria": "highly satisfactory",
    "satisfatorio": "satisfactory",
    "satisfatoria": "satisfactory",
    "moderadamente satisfatorio": "moderately satisfactory",
    "moderadamente satisfatoria": "moderately satisfactory",
    "insatisfatorio": "unsatisfactory",
    "insatisfatoria": "unsatisfactory",
    "moderadamente insatisfatorio": "moderately unsatisfactory",
    "moderadamente insatisfatoria": "moderately unsatisfactory",
    "muito insatisfatorio": "highly unsatisfactory",
    "muito insatisfatoria": "highly unsatisfactory",
    "excelente": "highly satisfactory",
    "muito bom": "highly satisfactory",
    "muito boa": "highly satisfactory",
    "bom": "satisfactory",
    "boa": "satisfactory",
    "razoavel": "moderately satisfactory",
    "regular": "moderately satisfactory",
    "fraco": "unsatisfactory",
    "fraca": "unsatisfactory",
    "muito fraco": "highly unsatisfactory",
    "muito fraca": "highly unsatisfactory",
    "atingido": "satisfactory",
    "atingida": "satisfactory",
    "nao atingido": "unsatisfactory",
    "nao atingida": "unsatisfactory",
    "alcancado": "satisfactory",
    "alcancada": "satisfactory",
    "nao alcancado": "unsatisfactory",
    "nao alcancada": "unsatisfactory",
    # -------------------------
    # German (after accent stripping)
    # -------------------------
    "sehr gut": "highly satisfactory",
    "ausgezeichnet": "highly satisfactory",
    "exzellent": "highly satisfactory",
    "hervorragend": "highly satisfactory",
    "gut": "satisfactory",
    "zufriedenstellend": "satisfactory",
    "eher gut": "moderately satisfactory",
    "teilweise zufriedenstellend": "moderately satisfactory",
    "mittel": "moderately satisfactory",
    "durchschnittlich": "moderately satisfactory",
    "akzeptabel": "moderately satisfactory",
    "unzureichend": "unsatisfactory",
    "schwach": "unsatisfactory",
    "mangelhaft": "unsatisfactory",
    "schlecht": "unsatisfactory",
    "sehr schlecht": "highly unsatisfactory",
    "ungenugend": "highly unsatisfactory",
    "erreicht": "satisfactory",
    "nicht erreicht": "unsatisfactory",
    "ziel erreicht": "satisfactory",
    "ziel nicht erreicht": "unsatisfactory",
    "stufe 2: erfolgreich": "satisfactory",
    "level 2 successful": "satisfactory",
    "sehr erfolgreich": "highly satisfactory",
    "erfolgreich": "satisfactory",
    "nicht erfolgreich": "unsatisfactory",
    # -------------------------
    # Expectations language (your rule)
    # -------------------------
    "significantly exceeded expectations": "highly satisfactory",
    "substantially exceeded expectations": "highly satisfactory",
    "far exceeded expectations": "highly satisfactory",
    "exceeded expectations": "highly satisfactory",
    "exceeding expectations": "highly satisfactory",
    "meeting expectations": "satisfactory",
    "met expectations": "satisfactory",
    "below expectations": "moderately unsatisfactory",
    "did not meet expectations": "unsatisfactory",
    "not meeting expectations": "unsatisfactory",
    "failed to meet expectations": "unsatisfactory",
    "significantly exceeded objective": "highly satisfactory",
    "exceeded objective": "highly satisfactory",
    "steadily advancing, achieving remarkable results": "highly satisfactory",
    "achieved objective": "satisfactory",
    "did not achieve objective": "unsatisfactory",
    # -------------------------
    # Status lights / tracking
    # -------------------------
    "green": "satisfactory",
    "amber": "moderately satisfactory",
    "yellow": "moderately satisfactory",
    "red": "unsatisfactory",
    "on track": "satisfactory",
    "fully implemented": "satisfactory",
    "on-track": "satisfactory",
    "on course": "satisfactory",
    "ahead of track": "highly satisfactory",
    "off track": "unsatisfactory",
    "off-track": "unsatisfactory",
    "delayed": "moderately unsatisfactory",
    "behind schedule": "moderately unsatisfactory",
    # -------------------------
    # Binary-ish / polarity words
    # -------------------------
    "achieved": "satisfactory",
    "not achieved": "unsatisfactory",
    "positive": "satisfactory",
    "negative": "unsatisfactory",
    "overall successful": "satisfactory",  # usually from XM-DAC binary prompt
    "overall unsuccessful": "moderately unsatisfactory",  # usually from XM-DAC binary prompt
    # -------------------------
    # Free-text performance (your rule: no neutral middle)
    # -------------------------
    "mixed performance": "moderately satisfactory",
    "some improvements": "moderately satisfactory",
    "moderately successful": "moderately satisfactory",
    "moderate": "moderately satisfactory",
    "partially successful": "moderately satisfactory",
    "partly successful": "moderately satisfactory",
    "relatively satisfying": "moderately satisfactory",
    "successfully": "moderately satisfactory",
    "needs improvement": "moderately unsatisfactory",
    "less than successful": "moderately unsatisfactory",
    "less than effective": "moderately unsatisfactory",
    "limited progress": "moderately unsatisfactory",
    "insufficient progress": "unsatisfactory",
    "nearly all indicators are satisfactorily achieved": "satisfactory",
    "very effectively though there is surely room for improvements": "satisfactory",
    "well": "satisfactory",
    "performed well": "satisfactory",
    "good": "satisfactory",
    "fair": "moderately satisfactory",
    "very effectively": "highly satisfactory",
    "excellent performance": "highly satisfactory",
    "excellent position": "highly satisfactory",
    "excellently": "highly satisfactory",
    "extremely well": "highly satisfactory",
    "very positive": "highly satisfactory",
    "extraordinary benefits": "highly satisfactory",
    "strong, positive impact": "highly satisfactory",
    "moving ahead and on track to achieving its objectives": "satisfactory",
    "a success": "satisfactory",
    "successful": "satisfactory",
    "highly successful": "highly satisfactory",
    "unsuccessful": "unsatisfactory",
    "highly unsuccessful": "highly unsatisfactory",
    "very successful": "highly satisfactory",
    "moderately unsuccessful": "moderately unsatisfactory",
    "very unsuccessful": "highly unsatisfactory",
    "mostly unsuccessful": "unsatisfactory",
    "disappointing": "unsatisfactory",
    "poor": "unsatisfactory",
    "poorly": "unsatisfactory",
    "weak": "unsatisfactory",
    "very weak": "highly unsatisfactory",
    "strong": "satisfactory",
    "substantial progress": "satisfactory",
    "very high level": "highly satisfactory",
    "largely exceeded expectations, representing very impressive value for money": "highly satisfactory",
    "a - meets expectations": "satisfactory",
    "a - meets expectations": "satisfactory",
    "resultado satisfactorio": "satisfactory",
}

# keep a pointer to the original (base) function so we can override safely

_DFID_GRADE_TO_0_5 = {
    "a++": 5.0,  # substantially exceeded expectations
    "a+": 4.5,  # exceeded expectations
    "a": 4.0,  # met expectations
    "b": 1.0,  # did not meet expectations (your rule)
    "c": 0.0,  # substantially did not meet
}


def _extract_numbers(x) -> list[float]:
    s = "" if x is None else str(x)
    s = s.replace(",", ".")
    return [float(m) for m in re.findall(r"(?<!\w)[+-]?\d+(?:\.\d+)?", s)]


def _coerce_num(x) -> Optional[float]:
    ns = _extract_numbers(x)
    return ns[0] if ns else None


def _parse_percent(x) -> Optional[float]:
    s = "" if x is None else str(x)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", s)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def _is_dfid_grade_family(v: str, mn: str, mx: str, activity_id=None) -> bool:
    if activity_id is None:
        return False
    aid = str(activity_id).upper()
    g = (v.split() or [""])[0]
    return aid.startswith("GB-") and g in _DFID_GRADE_TO_0_5  # catches GB-GOV-*, etc.


# add a few missing aliases (exact-match)
_ALIAS_TO_CANON.update(
    {
        "inadequate": "unsatisfactory",
        "acceptable": "moderately satisfactory",
        "impressive results": "satisfactory",
        "satisfied with most": "moderately satisfactory",
        "good overall progress": "moderately satisfactory",
        "good value for money": "satisfactory",
        "erfullt": "satisfactory",
        "rather successful": "moderately satisfactory",
        "eher erfolgreich": "moderately satisfactory",
        "parcialmente exitoso": "moderately satisfactory",
        "altamente exitoso": "highly satisfactory",
        "tres bon resultat": "highly satisfactory",
        "parcialmente insatisfactorio": "moderately unsatisfactory",
        "parcialmente insatisfactoria": "moderately unsatisfactory",
    }
)


def get_from_number(rating_value, max_rating, min_rating, activity_id):
    # handle "XX/YY" (arbitrary denominator) -> 0..5
    s = "" if rating_value is None else str(rating_value).strip()
    m = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*/\s*([0-9]+(?:[.,][0-9]+)?)\s*$", s)
    if m:
        num = float(m.group(1).replace(",", "."))
        den = float(m.group(2).replace(",", "."))
        if den != 0 and (0 <= num <= den):
            return 5 * (num / den)

    if _is_number(rating_value) and _is_number(max_rating) and _is_number(min_rating):
        lo = float(min_rating)
        hi = float(max_rating)
        v = float(rating_value)

        if hi == lo:
            return None
        if hi < lo:
            lo, hi = hi, lo

        if not (lo <= v <= hi):
            return None

        score = 5 * (v - lo) / (hi - lo)  # default: higher = better

        invert = (
            activity_id is not None
            and str(activity_id).startswith("DE-1")
            and lo == 1.0
            and hi == 6.0  # assume it's always inverted regardless of lo/high if de-1
        )
        if invert:
            score = 5 - score  # DE-1*: lower = better (1 best, 6 worst)

        return score


def get_success_measure_from_rating_value_wrapped(
    rating_value, min_rating=None, max_rating=None, activity_id=None
):

    v = _norm_text(rating_value)
    mn = _norm_text(min_rating)
    mx = _norm_text(max_rating)
    if activity_id.startswith("DE-1"):
        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s+von\s+16\s+punkten", v)
        if m:
            x = float(m.group(1).replace(",", "."))
            if 0 <= x <= 16:
                return 5 * (x / 16)

        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s+out\s+of\s+16\s+points", v)
        if m:
            x = float(m.group(1).replace(",", "."))
            if 0 <= x <= 16:
                return 5 * (x / 16)

        # replace things like "level 4: satisfactory"
        m = re.match(r"level\s+.\s*:\s*(.+?)\s*$", v, flags=re.IGNORECASE)
        if m:
            v = m.group(1).strip()

        # replace things like "nivel 4: satisfactorio"
        m = re.match(r"nivel\s+.\s*:\s*(.+?)\s*$", v, flags=re.IGNORECASE)
        if m:
            v = m.group(1).strip()

        v = v.split(" als ")[-1].strip()

        lo = _coerce_num(min_rating)
        hi = _coerce_num(max_rating)
        nums = _extract_numbers(v)

    pct = _parse_percent(v)
    if pct is not None:
        judged_percent_multiplied_by_5 = 5.0 * max(0.0, min(100.0, pct)) / 100.0
        if VERBOSE:
            print(
                f"Rating as number: {judged_percent_multiplied_by_5} from {rating_value}, min: {mn}, max: {mx}, act: {activity_id}"
            )

        return judged_percent_multiplied_by_5

    # 2) Normalize DFID-ish letter variants like "a-plus" -> "a+"
    v = (
        v.replace("a-plus", "a+")
        .replace("a plus", "a+")
        .replace("a-", "a-")
        .replace("a -", "a-")
    )
    v = (
        v.replace("b-plus", "b+")
        .replace("b plus", "b+")
        .replace("b-", "b-")
        .replace("b -", "b-")
    )
    v = (
        v.replace("c-plus", "c+")
        .replace("c plus", "c+")
        .replace("c-", "c-")
        .replace("c -", "c-")
    )
    v = (
        v.replace("d-plus", "d+")
        .replace("d plus", "d+")
        .replace("d-", "d-")
        .replace("d -", "d-")
    )
    v = (
        v.replace("f-plus", "f+")
        .replace("f plus", "f+")
        .replace("f-", "f-")
        .replace("f -", "f-")
    )
    if v.startswith("a") and "met expectation" in v:
        v = "a"
    if v.startswith("a+") and "exceed" in v:
        v = "a+"

    # 0) DFID/FCDO Annual Review grade family FIRST (so it overrides base letter-grade mapping)
    if _is_dfid_grade_family(v, mn, mx, activity_id=activity_id):
        g = v.split()[0]  # handles e.g. "a - meets expectations" -> "a"
        if g in _DFID_GRADE_TO_0_5:
            rating = _DFID_GRADE_TO_0_5[g]
            # print(f"Rating: {rating} from {rating_value}| Min: {min_rating} | Max: {max_rating} | GB-1")
            return rating

    ### only thing i change
    # v_no_parens = re.sub(r"\s*\(.*?\)\s*", " ", v).strip()
    # m = re.search(r"\(([^()]*)\)", v)
    # v_inside_parens = m.group(1).strip() if m else v

    # for attempt in (v, v_no_parens, v_inside_parens):
    ### end only thing i change
    raw = "" if rating_value is None else str(rating_value)

    cand_raw = raw
    cand_no_parens = re.sub(r"\([^)]*\)", " ", raw).strip()
    m = re.search(r"\(([^()]*)\)", raw)
    cand_in_parens = m.group(1).strip() if m else ""

    parts = re.split(
        r"\s+als\s+|\s+=\s+|\s*:\s*|\s*-\s*|\s*-\s*|\s*--\s*", raw, maxsplit=1
    )
    cand_before_split = parts[0].strip() if len(parts) > 1 else ""
    cand_after_split = parts[1].strip() if len(parts) > 1 else ""

    for cand in (
        cand_raw,
        cand_no_parens,
        cand_in_parens,
        cand_before_split,
        cand_after_split,
    ):
        if not cand:
            continue
        attempt = _norm_text(cand)

        if activity_id.startswith("XI-IATI-IA"):
            # "EVALUABLE (SCORE 6.8)" -> 0..10 mapped to 0..5
            m = re.search(
                r"score\s*([0-9]+(?:[.,][0-9]+)?)",
                str(rating_value),
                flags=re.IGNORECASE,
            )
            if m:
                sc = float(m.group(1).replace(",", "."))
                return 5.0 * max(0.0, min(10.0, sc)) / 10.0

            # "0.81 - SATISFACTORIO" -> prob 0..1 mapped to 0..5
            m = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*[-:]", str(rating_value))
            if m:
                p = float(m.group(1).replace(",", "."))
                if 0.0 <= p <= 1.0:
                    return 5.0 * p

            IADB_RATINGS = {
                "very probable": 5.0,
                "muy probable": 5.0,
                "probable": 5.0 * 2 / 3,
                "provavel": 5.0 * 2 / 3,  # "Provavel (P)" -> _norm_text -> "provavel"
                "low probability": 5.0 * 1 / 3,
                "poco probable": 5.0
                * 1
                / 3,  # "Poco Probable (PP)" -> _norm_text -> "poco probable"
                "improbable": 0.0,
            }

            result = IADB_RATINGS.get(attempt)
            if result is not None:
                return result

        # 4) Handle known aliases/translations/free-text
        canon = _ALIAS_TO_CANON.get(attempt)
        if canon is not None:
            rating = float(_CANON_LABEL_TO_0_5[canon])
            # print(f"Rating: {rating} from {rating_value}| Min: {min_rating} | Max: {max_rating} | canonical")
            return rating

    for cand in (
        cand_raw,
        cand_no_parens,
        cand_in_parens,
        cand_before_split,
        cand_after_split,
    ):
        if not cand:
            continue
        attempt = _norm_text(cand)
        # # 5) Handle strings like "3 - tentative evidence..." with Level 1..5 endpoints
        v0 = get_success_measure_from_rating_value(
            attempt, min_rating, max_rating, activity_id=activity_id
        )
        if v0 is not None:
            # print(f"Rating: {v0} from {rating_value}| Min: {min_rating} | Max: {max_rating}")
            return v0

        # now let's try some backups "contains"

        if "highly successful" in attempt:
            return 5
        if "highly unsuccessful" in attempt:
            return 0
        if "moderately successful" in attempt:
            return 4
        if "moderately unsuccessful" in attempt:
            return 2

        if attempt.startswith("successfully"):
            return 4

        if attempt.startswith("satisfactorily"):
            return 4

        if attempt.endswith("very successful") and not "not" in attempt:
            return 5

    for cand in (
        cand_raw,
        cand_no_parens,
        cand_in_parens,
        cand_before_split,
        cand_after_split,
    ):
        if not cand:
            continue
        attempt = _norm_text(cand)

        number_parsed = get_from_number(attempt, mn, mx, activity_id)
        if number_parsed is not None:
            if VERBOSE:
                print(
                    f"Rating as number: {number_parsed} from {rating_value}, min: {mn}, max: {mx}, act: {activity_id}"
                )
            return number_parsed

    #     # plain numeric already 0..5
    #     if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", attempt):
    #         x = float(attempt)
    #         if 0.0 <= x <= 5.0:
    #             return x

    #     # "3 - Tentative evidence ..." etc.
    #     lead = _maybe_parse_leading_int(attempt)
    #     if lead is not None and 0 <= lead <= 5:
    #         return float(lead)

    #     if "satisfactorily" in attempt:
    #         return 4.0

    if v.endswith("; successful"):
        return 4
    if v.endswith(", successful"):
        return 4

    return None


def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


def pick_start_date(row: pd.Series):
    for c in ["actual_start_date", "original_planned_start_date", "txn_first_date"]:
        if c in row and pd.notna(row[c]):
            return row[c]
    return pd.NaT


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------


def load_grades(pattern):
    """Load all grades files matching pattern"""
    grades_data = {}

    for filepath in glob.glob(pattern):
        feature_name = (
            Path(filepath).stem.replace("outputs_", "").replace("_grades", "")
        )
        print(f"Loading {feature_name}...")

        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)
                activity_id = data["activity_id"]

                response_text = data.get("response_text") or data.get(
                    "response", {}
                ).get("content", "")
                grade = None
                if "GRADE:" in response_text:
                    try:
                        grade = int(response_text.split("GRADE:")[1].strip())
                    except ValueError:
                        grade = None

                if activity_id not in grades_data:
                    grades_data[activity_id] = {}
                grades_data[activity_id][feature_name] = grade

    return pd.DataFrame.from_dict(grades_data, orient="index")


from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import re
import numpy as np
import pandas as pd

_NUM_RE = re.compile(r"\d+")


def load_targets_context_maps_features(
    path: Path,
    *,
    sector_levels: Optional[List[str]] = None,
    drop_sector: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads ../../data/outputs_targets_context_maps.jsonl and returns a DF indexed by activity_id with:
      - umap2_x, umap2_y  (from umap_2d only)
      - text_len_chars   (len of concatenated `text` across ALL rows for that activity_id)
      - targets_number_count (count of numbers in concatenated `text` across ALL rows for that activity_id)
      - sector one-hot dummies with 1 dropped (5 sectors -> 4 cols) to reduce colinearity

    Safe for multiple rows per activity_id: it accumulates text; sector/umap2 use first valid value seen.
    """
    path = Path(path)

    sector_by_aid: Dict[str, Optional[str]] = {}
    umap2_by_aid: Dict[str, List[float]] = {}
    umap3_by_aid: Dict[str, List[float]] = {}
    umap4_by_aid: Dict[str, List[float]] = {}
    text_parts_by_aid: Dict[str, List[str]] = {}
    sector_dist_by_aid: Dict[str, float] = {}
    country_dist_by_aid: Dict[str, float] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj: Dict[str, Any] = json.loads(s)

            aid = str(obj.get("activity_id", "")).strip()
            if not aid:
                continue

            # sector_distance (first finite wins)
            if aid not in sector_dist_by_aid:
                v = obj.get("sector_distance", None)
                if v is not None:
                    fv = float(v)
                    if np.isfinite(fv):
                        sector_dist_by_aid[aid] = fv

            # country_distance (first finite wins)
            if aid not in country_dist_by_aid:
                v = obj.get("country_distance", None)
                if v is not None:
                    fv = float(v)
                    if np.isfinite(fv):
                        country_dist_by_aid[aid] = fv

            # sector (first non-empty wins)
            if aid not in sector_by_aid or sector_by_aid[aid] is None:
                sec = obj.get("sector", None)
                if isinstance(sec, str) and sec.strip():
                    sector_by_aid[aid] = sec.strip()

            # umap_2d (first valid wins)
            if aid not in umap2_by_aid:
                u2 = obj.get("umap_2d", None)
                if isinstance(u2, (list, tuple)) and len(u2) == 2:
                    umap2_by_aid[aid] = [float(u2[0]), float(u2[1])]

            # umap_3d (first valid wins)
            if aid not in umap3_by_aid:
                u3 = obj.get("umap_3d", None)
                if isinstance(u3, (list, tuple)) and len(u3) == 3:
                    umap3_by_aid[aid] = [float(u3[0]), float(u3[1]), float(u3[2])]

            # umap_4d (first valid wins)
            if aid not in umap4_by_aid:
                u4 = obj.get("umap_4d", None)
                if isinstance(u4, (list, tuple)) and len(u4) == 4:
                    umap4_by_aid[aid] = [
                        float(u4[0]),
                        float(u4[1]),
                        float(u4[2]),
                        float(u4[3]),
                    ]

    # build DF index from all ids we saw
    all_aids = sorted(
        set(sector_by_aid)
        | set(umap2_by_aid)
        | set(umap3_by_aid)
        | set(umap4_by_aid)
        | set(text_parts_by_aid)
        | set(sector_dist_by_aid)
        | set(country_dist_by_aid)
    )
    df = pd.DataFrame(index=pd.Index(all_aids, name="activity_id"))

    df["sector_distance"] = [
        sector_dist_by_aid.get(aid, np.nan) for aid in df.index.astype(str)
    ]
    df["country_distance"] = [
        country_dist_by_aid.get(aid, np.nan) for aid in df.index.astype(str)
    ]

    # umap2 -> x/y
    df["umap2_x"] = [
        umap2_by_aid.get(aid, [np.nan, np.nan])[0] for aid in df.index.astype(str)
    ]
    df["umap2_y"] = [
        umap2_by_aid.get(aid, [np.nan, np.nan])[1] for aid in df.index.astype(str)
    ]
    df["umap3_x"] = [
        umap3_by_aid.get(aid, [np.nan, np.nan, np.nan])[0]
        for aid in df.index.astype(str)
    ]
    df["umap3_y"] = [
        umap3_by_aid.get(aid, [np.nan, np.nan, np.nan])[1]
        for aid in df.index.astype(str)
    ]
    df["umap3_z"] = [
        umap3_by_aid.get(aid, [np.nan, np.nan, np.nan])[2]
        for aid in df.index.astype(str)
    ]
    df["umap4_x"] = [
        umap4_by_aid.get(aid, [np.nan, np.nan, np.nan, np.nan])[0]
        for aid in df.index.astype(str)
    ]
    df["umap4_y"] = [
        umap4_by_aid.get(aid, [np.nan, np.nan, np.nan, np.nan])[1]
        for aid in df.index.astype(str)
    ]
    df["umap4_z"] = [
        umap4_by_aid.get(aid, [np.nan, np.nan, np.nan, np.nan])[2]
        for aid in df.index.astype(str)
    ]
    df["umap4_w"] = [
        umap4_by_aid.get(aid, [np.nan, np.nan, np.nan, np.nan])[3]
        for aid in df.index.astype(str)
    ]

    # df["text"] = [text_by_aid.get(aid) for aid in df.index.astype(str)]
    # df["text_len_chars"] = [len(text_by_aid.get(aid).get("text")) for aid in df.index.astype(str) if text_by_aid.get(aid) is not N]

    # targets_number_count = count of numbers in ALL text
    # df["targets_number_count"] = [
    #     len(_NUM_RE.findall(text_by_aid.get(aid).get("text")))
    #     for aid in df.index.astype(str)
    # ]
    # sector one-hot (drop baseline to reduce colinearity)
    sector_s = pd.Series(
        {aid: sector_by_aid.get(aid) for aid in df.index.astype(str)},
        index=df.index,
        name="env_cat",
    )

    # # CRASH if any missing sector
    # if sector_s.isna().any():
    #     bad = sector_s[sector_s.isna()].index[:20].tolist()
    #     raise ValueError(f"Missing sector for {int(sector_s.isna().sum())} activity_ids (examples: {bad})")

    if sector_levels is None:
        sector_levels = sorted(sector_s.dropna().unique().tolist())

    # IMPORTANT: keep index by wrapping as Series BEFORE get_dummies
    sector_ser = pd.Series(
        pd.Categorical(sector_s, categories=sector_levels),
        index=df.index,
        name="env_cat",
    )
    sector_dum = pd.get_dummies(sector_ser, prefix="env_cat", dtype=int)

    # (optional) if you want safe column names, keep this; otherwise delete these 2 lines
    sector_dum = sector_dum.rename(columns=lambda c: re.sub(r"[^0-9A-Za-z_]+", "_", c))
    if drop_sector is None and sector_levels:
        drop_sector = sector_levels[0]

    if drop_sector is not None:
        drop_col = "env_cat" + re.sub(r"[^0-9A-Za-z_]+", "_", drop_sector)
        if drop_col in sector_dum.columns:
            sector_dum = sector_dum.drop(columns=[drop_col])

    out = df.join(sector_dum, how="left")

    # print("\n[tc_maps] out.shape:", out.shape)
    # print("[tc_maps] columns:", list(out.columns))

    # NaN rates (top 25)
    nan_pct = (out.isna().mean() * 100).sort_values(ascending=False)
    # print("\n[tc_maps] NaN % (top 25):")
    # print(nan_pct.head(25).to_string())

    # sector diagnostics
    # print("\n[tc_maps] sector label counts:")
    # print(sector_s.value_counts(dropna=False).to_string())

    sec_cols = [c for c in out.columns if c.startswith("sector_")]
    # print("\n[tc_maps] sector dummy sums:")
    # print(out[sec_cols].sum().sort_values(ascending=False).to_string())

    # 5 full random rows
    n = min(5, len(out))
    # print("\n[tc_maps] sample rows:")
    # print(out.sample(n=n, random_state=0).to_string())

    # crash if columns duplicated (useful if sanitization accidentally collides)
    dup_cols = out.columns[out.columns.duplicated()].unique().tolist()
    if dup_cols:
        raise ValueError(f"[tc_maps] Duplicate columns produced: {dup_cols}")

    return out


def load_ratings(filepath: str) -> pd.Series:
    """
    Load numeric ratings from merged_overall_ratings.jsonl.

    "Good" rating criteria:
      - rating object comes from:
          * response_text (JSON string or dict), else
          * from_gemini.overall_rating (dict)
      - rating_value is non-empty
      - description != 'NO RATING AVAILABLE' (case-insensitive)
    """
    ratings = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except Exception:
                continue

            aid = str(obj.get("activity_id") or "").strip()
            if not aid:
                continue

            rating_obj = None

            # 1) response_text: JSON string or dict
            rt = obj.get("response_text")
            if isinstance(rt, str) and rt:
                try:
                    rating_obj = json.loads(rt)
                except Exception:
                    rating_obj = None
            elif isinstance(rt, dict):
                rating_obj = rt

            # 2) fallback: from_gemini.overall_rating
            if rating_obj is None:
                fg = obj.get("from_gemini")
                if isinstance(fg, dict):
                    maybe_overall = fg.get("overall_rating")
                    if isinstance(maybe_overall, dict):
                        rating_obj = maybe_overall

            if not isinstance(rating_obj, dict):
                continue

            desc = str(rating_obj.get("description") or "").strip()
            rating_value = str(rating_obj.get("rating_value") or "").strip()
            if not rating_value:
                continue
            if desc.upper() == "NO RATING AVAILABLE":
                continue

            rating_min = rating_obj.get("min")
            rating_max = rating_obj.get("max")

            numeric_rating = get_success_measure_from_rating_value_wrapped(
                rating_value, rating_min, rating_max, activity_id=aid
            )
            if VERBOSE:
                print(
                    f"Rating as number: {numeric_rating} from {rating_value}, min: {rating_min}, max: {rating_max}, act: {aid}"
                )

            if numeric_rating is not None:
                ratings[aid] = numeric_rating

    return pd.Series(ratings, name="rating")


def load_activity_scope(filepath):
    """Load activity scope from CSV and map to numeric codes."""
    df = pd.read_csv(filepath, usecols=["activity_id", "activity_scope"])
    df["activity_scope"] = df["activity_scope"].astype(str).str.strip().str.lower()
    df["activity_scope"] = df["activity_scope"].map(ACTIVITY_SCOPES)
    return df.set_index("activity_id")


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


def load_gdp_percap(filepath):
    """Load activity scope from CSV and map to numeric codes."""
    df = pd.read_csv(filepath, usecols=["activity_id", "gdp_percap"])
    df["gdp_percap"] = df["gdp_percap"].astype(float)
    df["gdp_percap"] = np.log(df["gdp_percap"])
    return df.set_index("activity_id")


def load_world_bank_indicators(filepath):
    """Load activity scope from CSV and map to numeric codes."""
    df = pd.read_csv(
        filepath,
        usecols=[
            "activity_id",
            "cpia_score",
            "wgi_control_of_corruption_est",
            "wgi_government_effectiveness_est",
            "wgi_political_stability_est",
            "wgi_regulatory_quality_est",
            "wgi_rule_of_law_est",
        ],
    )
    df["cpia_score"] = df["cpia_score"].astype(float)
    df["wgi_control_of_corruption_est"] = df["wgi_control_of_corruption_est"].astype(
        float
    )
    df["wgi_government_effectiveness_est"] = df[
        "wgi_government_effectiveness_est"
    ].astype(float)
    df["wgi_political_stability_est"] = df["wgi_political_stability_est"].astype(float)
    df["wgi_regulatory_quality_est"] = df["wgi_regulatory_quality_est"].astype(float)
    df["wgi_rule_of_law_est"] = df["wgi_rule_of_law_est"].astype(float)
    return df.set_index("activity_id")


def load_is_completed(filepath):
    """Load activity completion flag from CSV."""
    df = pd.read_csv(filepath, usecols=["activity_id", "status_code"])
    status_code = df["status_code"].astype(int)

    # Description
    # 1   Pipeline/identification     The activity is being scoped or planned
    # 2   Implementation  The activity is currently being implemented
    # 3   Finalisation    Physical activity is complete or the final disbursement has been made, but the activity remains open pending financial sign off or M&E
    # 4   Closed  Physical activity is complete or the final disbursement has been made.
    # 5   Cancelled   The activity has been cancelled
    # 6   Suspended   The activity has been temporarily suspended

    # 3 = Finalisation, 4 = Closed , 5 = cancelled -> considered "completed"
    # we already removed "pipeline" activities
    df["is_completed"] = status_code.isin([3, 4, 5]).astype(int)

    return df.set_index("activity_id")[["is_completed"]]


import pandas as pd


def load_implementing_org_type(filepath: str) -> pd.DataFrame:
    """
    From info_for_activity_forecasting CSV:
      implementing_org_type in {"government","ngo","other"} (string)

    Returns two one-hot-ish cols:
      - is_government_impl
      - is_ngo_impl
    "other" (and unknowns) -> both 0.
    """
    df = pd.read_csv(filepath, usecols=["activity_id", "implementing_org_type"])

    s = df["implementing_org_type"].astype(str).str.strip().str.lower()

    s = s.where(s.isin(["govermental", "ngo", "other"]), "other")

    out = pd.DataFrame(
        {
            "activity_id": df["activity_id"].astype(str),
            "is_government_impl": (s == "govermental").astype(int),
            "is_ngo_impl": (s == "ngo").astype(int),
        }
    )

    return out.set_index("activity_id")


def data_sector_clusters(filepath: str, train_activity_ids=None) -> pd.DataFrame:
    """
    Load sector cluster allocations from finance sectors disbursements file.
    Returns DataFrame with activity_id as index and sector_hhi, n_sectors, sector_cluster_* columns.
    """

    finance_file = Path(filepath)

    # Use cache for embeddings to speed up re-runs
    cache_file = finance_file.parent / "sector_label_embeddings.pkl"
    # Fallback: activities where PDF extraction found no sector data, retried using finance text
    fallback_file = (
        finance_file.parent / "outputs_finance_sectors_from_finance_text.jsonl"
    )

    df = process_finance_sectors_to_clusters(
        finance_file=finance_file,
        fallback_file=fallback_file,
        embeddings_cache=cache_file,
        n_clusters=15,
        force_recompute=False,
        train_activity_ids=train_activity_ids,
    )

    if df is None:
        print(
            "WARNING: could not process finance sectors to clusters, returning empty DataFrame"
        )
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=["sector_hhi", "n_sectors"]).set_index(
            pd.Index([], name="activity_id")
        )

    print(f"Loaded {len(df)} activities with sector cluster data")
    return df


def restrict_to_reporting_orgs_exact(df, KEEP_REPORTING_ORGS) -> pd.DataFrame:
    BMZ_A = "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); Federal Ministry for Economic Cooperation and Development (BMZ)"
    BMZ_B = "Federal Ministry for Economic Cooperation and Development (BMZ); Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ)"

    s = df["reporting_orgs"].fillna("").astype(str).str.strip()
    s = s.replace(BMZ_B, BMZ_A)
    mask = s.isin(KEEP_REPORTING_ORGS)
    out = df.loc[mask].copy()
    out["reporting_orgs"] = s.loc[mask]
    return out


def add_similarity_features(
    data,
    info_csv_path,
    keep_reporting_orgs,
) -> tuple[pd.DataFrame, list[str]]:
    print("\nLoading info_for_activity_forecasting for similarity-style features...")
    info_df = pd.read_csv(info_csv_path)

    for col in ["txn_first_date", "actual_start_date", "original_planned_start_date"]:
        info_df[col] = pd.to_datetime(info_df.get(col), errors="coerce")

    info_df["start_date"] = info_df.apply(pick_start_date, axis=1)

    # align indexes
    info_df["activity_id"] = info_df["activity_id"].astype(str)
    info_df = info_df.set_index("activity_id")
    data.index = data.index.astype(str)

    info_df = info_df.loc[info_df.index.intersection(data.index)]

    rep_org_vocab = {org: i for i, org in enumerate(keep_reporting_orgs)}
    BMZ_A = "Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ); Federal Ministry for Economic Cooperation and Development (BMZ)"
    BMZ_B = "Federal Ministry for Economic Cooperation and Development (BMZ); Bundesministerium für wirtschaftliche Zusammenarbeit und Entwicklung (BMZ)"

    s = info_df["reporting_orgs"].fillna("").astype(str).str.strip()
    s = s.replace(BMZ_B, BMZ_A)

    # map directly from the raw reporting_orgs string (exact match)
    data["rep_org_main_idx"] = s.map(rep_org_vocab).reindex(data.index).astype("Int64")
    # one-hot with stable rep_org_0..rep_org_{n-1} columns
    rep_org_ohe = pd.get_dummies(data["rep_org_main_idx"], dtype=int)
    rep_org_ohe = rep_org_ohe.reindex(
        columns=range(len(keep_reporting_orgs)), fill_value=0
    )
    rep_org_ohe.columns = [f"rep_org_{i}" for i in range(len(keep_reporting_orgs))]

    data = pd.concat([data, rep_org_ohe], axis=1)
    all_rep_cols = [f"rep_org_{i}" for i in range(len(keep_reporting_orgs))]

    # keep raw org string around for inspection/debug
    data["reporting_orgs"] = s.reindex(data.index)

    sim_feature_cols = all_rep_cols
    return data, sim_feature_cols


def add_dates_to_dataframe(X, INFO_FOR_ACTIVITY_FORECASTING):
    dates_df = pd.read_csv(
        INFO_FOR_ACTIVITY_FORECASTING,
        usecols=[
            "activity_id",
            "reporting_orgs",
            "txn_first_date",
            "actual_start_date",
            "original_planned_start_date",
            "original_planned_close_date",
        ],
    )

    for col in [
        "txn_first_date",
        "actual_start_date",
        "original_planned_start_date",
        "original_planned_close_date",
    ]:
        dates_df[col] = pd.to_datetime(dates_df[col], errors="coerce")

    # duration in YEARS
    dates_df["planned_duration"] = (
        dates_df["original_planned_close_date"]
        - dates_df["original_planned_start_date"]
    ).dt.days / 365.25

    # <- alias for the splitter (it wants this exact name)
    # dates_df["planned_duration"] = dates_df["planned_duration"]

    # <- splitter wants rep_org
    # dates_df["rep_org"] = dates_df["reporting_orgs"]#.map(normalize_rep_org)

    dates_df["start_date"] = dates_df.apply(pick_start_date, axis=1)

    dates_df = dates_df[["activity_id", "start_date", "planned_duration"]]
    # print(f"LEN DATA BEFORE DROP START DATE: {dates_df}")
    dates_df = dates_df.dropna(subset=["start_date"]).set_index("activity_id")
    # print(f"LEN DATA AFTER DROP START DATE: {dates_df}")

    common_ids = dates_df.index.intersection(X.index)
    dates_df = dates_df.loc[common_ids]

    return X.join(dates_df, how="inner")


import re

# Put your canonical labels here (whatever get_success_measure_from_rating_value expects)
CANON_LABELS = [
    "Highly Satisfactory",
    "Satisfactory",
    "Moderately Satisfactory",
    "Moderately Unsatisfactory",
    "Unsatisfactory",
    "Highly Unsatisfactory",
]

# Longest-first prevents matching "Satisfactory" inside "Moderately Satisfactory"
_LABEL_RE = re.compile(
    r"\b("
    + "|".join(map(re.escape, sorted(CANON_LABELS, key=len, reverse=True)))
    + r")\b",
    flags=re.IGNORECASE,
)

_MD_STRIP_RE = re.compile(r"[*_`]+")  # cheap markdown marker removal


def parse_last_line_label_after_forecast(content, record=None):
    """
    Content is long text; last non-empty line contains the label,
    e.g. "FORECAST: Moderately Satisfactory" or "4. **Forecast:** Successful".

    Returns a numeric rating on your canonical 0-5 scale
    using get_success_measure_from_rating_value.
    """
    text = str(content)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        print("ERROR: failed to parse last line result (no non-empty lines).")
        print("failed text:")
        pprint.pprint(lines)
        return None

    last = lines[-1]

    # 1) strip leading numbering / bullets like:
    #    "4. ", "4)", "4]", "- ", "* "
    line = re.sub(r"^\s*(?:[-*+]|\d+[\).\]])\s*", "", last)

    # 2) remove markdown bold/italic markers
    #    (this is aggressive, but you said you want to strip **, *, _)
    line = line.replace("**", "").replace("__", "")
    line = line.strip(" *_")

    # 3) find "forecast" (case-insensitive) and capture text after it
    #    allow optional : or dash after the word
    m = re.search(r"forecast\s*[:\----]?\s*(.+)$", line, flags=re.IGNORECASE)
    if not m:
        print("ERROR: last line does not contain 'FORECAST'")
        print("last line:", last)
        return None

    label = m.group(1).strip()

    # 4) strip trailing decoration/punctuation from the label
    #    e.g. "Successful.", "Successful**", "Successful_" -> "Successful"
    label = label.strip(" *.?_\"'`-")

    # Optionally, get min/max from record if present
    rating_min = None
    rating_max = None
    if isinstance(record, dict):
        rating_min = record.get("min")
        rating_max = record.get("max")

    activity_id = None
    if isinstance(record, dict):
        activity_id = record.get("activity_id")
    # print("activity id")
    # print(activity_id)
    numeric = get_success_measure_from_rating_value_wrapped(
        label,
        min_rating=rating_min,
        max_rating=rating_max,
        activity_id=activity_id,
    )

    if numeric is None:
        print("ERROR: failed to map forecast label to numeric rating")
        print("last line:", last)
        print("parsed label:", label)
        return None

    return numeric


# ---------------------------------------------------------------------------
# ENHANCED UNCERTAINTY FEATURES (Strategy 8)
# ---------------------------------------------------------------------------


def add_enhanced_uncertainty_features(data, feature_cols_llm=None):
    """
    Enhanced uncertainty/data quality features with individual missingness indicators.
    These features capture meta-information about data completeness and help the model
    calibrate its predictions based on input reliability.
    """
    # print("\n=== Adding Enhanced Uncertainty/Data Quality Features ===")

    # LLM features missingness
    llm_features = [
        "finance",
        "integratedness",
        "implementer_performance",
        "targets",
        "context",
        "risks",
        "complexity",
    ]
    llm_features = [f for f in llm_features if f in data.columns]

    if llm_features:
        data["llm_features_missing_count"] = data[llm_features].isna().sum(axis=1)
        data["llm_features_present_ratio"] = 1 - (
            data["llm_features_missing_count"] / len(llm_features)
        )

    # Governance features missingness
    gov_features = ["cpia_score"] + [c for c in data.columns if "wgi_" in c]
    if gov_features:
        data["governance_missing_count"] = data[gov_features].isna().sum(axis=1)

    # Overall feature completeness
    if feature_cols_llm:
        feature_cols_to_check = [c for c in feature_cols_llm if c in data.columns]
        if feature_cols_to_check:
            data["feature_completeness_ratio"] = 1 - (
                data[feature_cols_to_check].isna().sum(axis=1)
                / len(feature_cols_to_check)
            )

    # ---- Individual missingness indicators for key features ----

    # CPIA score missing indicator
    if "cpia_score" in data.columns:
        data["cpia_missing"] = data["cpia_score"].isna().astype(float)

    # Sector cluster missing indicator
    sector_cluster_cols = [c for c in data.columns if c.startswith("sector_cluster_")]
    if sector_cluster_cols:
        sector_data_present = (data[sector_cluster_cols].notna().any(axis=1)) | (
            data[sector_cluster_cols].sum(axis=1) > 0
        )
        data["sector_clusters_missing"] = (~sector_data_present).astype(float)

    # GDP per capita missing
    if "gdp_percap" in data.columns:
        data["gdp_percap_missing"] = data["gdp_percap"].isna().astype(float)

    # Planned expenditure missing
    if "planned_expenditure" in data.columns:
        data["planned_expenditure_missing"] = (
            data["planned_expenditure"].isna().astype(float)
        )

    # Planned duration missing
    if "planned_duration" in data.columns:
        data["planned_duration_missing"] = data["planned_duration"].isna().astype(float)

    # WGI indicators missing
    wgi_cols = [c for c in data.columns if "wgi_" in c]
    if wgi_cols:
        data["wgi_any_missing"] = data[wgi_cols].isna().any(axis=1).astype(float)

    # UMAP coordinates missing
    umap_cols = ["umap3_x", "umap3_y", "umap3_z"]
    umap_cols_present = [c for c in umap_cols if c in data.columns]
    if umap_cols_present:
        data["umap_missing"] = data[umap_cols_present].isna().any(axis=1).astype(float)

    n_features = 12  # 5 aggregate + 7 individual flags
    # print(f"  Added {n_features} enhanced uncertainty features")

    return data
