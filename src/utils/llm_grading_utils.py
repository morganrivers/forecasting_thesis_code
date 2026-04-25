"""
Utility functions for grading and analyzing LLM forecast outputs.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from feature_engineering import load_ratings
from scoring_metrics import mae, rmse, side_accuracy
from scoring_metrics import r2 as r2_metric

# ---------------------------------------------------------------------------
# LOADING AND BASIC PARSING
# ---------------------------------------------------------------------------


def load_jsonl_by_activity_id(
    path: Path, content_key: str = "response"
) -> dict[str, str]:
    """
    Load JSONL and return dict: activity_id -> text content.

    For forecasts: extract from response.content, response.text, or response_text
    For outcomes: extract from response.content, response.text, or response_text
    """
    out: dict[str, str] = {}

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            aid = str(data.get("activity_id") or "").strip()
            if not aid:
                continue

            resp = data.get("response")
            txt = ""
            if isinstance(resp, dict):
                txt = resp.get("content") or resp.get("text") or ""

            if not txt:
                txt = data.get("response_text") or ""

            txt = str(txt).strip()
            if txt and txt != "NO RESPONSE":
                out[aid] = txt

    return out


# ---------------------------------------------------------------------------
# GRADE EXTRACTION AND METRICS
# ---------------------------------------------------------------------------

GRADE_TO_PCT = {
    "A+": 97,
    "A": 95,
    "A-": 92,
    "B+": 88,
    "B": 85,
    "B-": 82,
    "C+": 78,
    "C": 75,
    "C-": 72,
    "D+": 68,
    "D": 65,
    "D-": 62,
    "F": 55,
}

GRADE_PATTERN = re.compile(r"GRADE:\s*([A-F][+-]?)(?=\s|$)", re.IGNORECASE)
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]

# Thesis grading_free_form: exact grade scale values
assert GRADE_TO_PCT["F"] == 55, "thesis: F=55"
assert GRADE_TO_PCT["A+"] == 97, "thesis: A+=97"
assert GRADE_TO_PCT["D-"] == 62, "thesis: D-=62"
assert GRADE_TO_PCT["D"] == 65, "thesis: D=65"
assert GRADE_TO_PCT["D+"] == 68, "thesis: D+=68"
assert GRADE_TO_PCT["C-"] == 72, "thesis: C-=72"
assert (
    len(GRADE_ORDER) == 13
), f"thesis: 13 letter grades in GRADE_ORDER, got {len(GRADE_ORDER)}"
assert set(GRADE_TO_PCT.keys()) == set(
    GRADE_ORDER
), "GRADE_TO_PCT keys must match GRADE_ORDER"
_pcts = [GRADE_TO_PCT[g] for g in GRADE_ORDER]
assert all(
    _pcts[i] > _pcts[i + 1] for i in range(len(_pcts) - 1)
), "grade percentages must be strictly decreasing from A+ to F"


def extract_forecast_ratings(path: Path, parser=None) -> pd.Series:
    """
    Extract numeric ratings (0-5) from forecast outputs.

    Uses parse_last_line_label_after_forecast by default if available,
    or a provided parser function.
    """
    if parser is None:
        try:
            from feature_engineering import (
                parse_last_line_label_after_forecast,
            )

            parser = parse_last_line_label_after_forecast
        except ImportError as err:
            raise ValueError(
                "Must provide parser function or have feature_engineering available"
            ) from err

    preds = {}

    if not path.exists():
        raise FileNotFoundError(f"Predictions JSONL not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            activity_id = data.get("activity_id")
            if activity_id is None:
                continue

            content = None
            resp = data.get("response")
            if isinstance(resp, dict):
                content = resp.get("content") or resp.get("text")

            if not content:
                content = data.get("response_text")

            if not content:
                continue

            value = parser(content, data)
            if value is None:
                continue

            preds[activity_id] = value

    return pd.Series(preds, name=path.stem)


# ---------------------------------------------------------------------------
# METRIC CALCULATIONS
# ---------------------------------------------------------------------------

# rmse, mae, side_accuracy imported from scoring_metrics above


def calculate_metrics(y_true, y_pred) -> dict:
    """Returns rmse, r2, mae, side_accuracy_3_5, n for finite (y_true, y_pred) pairs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    if len(y_true_f) == 0:
        return {
            "rmse": np.nan,
            "r2": np.nan,
            "mae": np.nan,
            "side_accuracy_3_5": np.nan,
            "n": 0,
        }

    return {
        "rmse": rmse(y_true_f, y_pred_f),
        "r2": r2_metric(y_true_f, y_pred_f),
        "mae": mae(y_true_f, y_pred_f),
        "side_accuracy_3_5": side_accuracy(y_true_f, y_pred_f, threshold=3.5),
        "n": len(y_true_f),
    }


# ---------------------------------------------------------------------------
# GROUND TRUTH LOADING
# ---------------------------------------------------------------------------


def load_ground_truth_ratings(ratings_path: Path) -> pd.Series:
    """
    Load ground truth ratings from JSONL file.

    Returns Series: activity_id -> numeric rating (0-5 scale)
    """
    ratings = load_ratings(str(ratings_path))
    return ratings
