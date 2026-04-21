"""
Leakage-risk activity IDs, derived from data/LEAKAGERISK.json.

EXCLUDE_TEST_LEAKAGE_RISK: when False, all leakage-handling code is a no-op.
Set exclude_test_leakage_risk to false in the JSON to use as a comparison baseline.

Each activity entry in LEAKAGERISK.json has the form:
  {
    "activity_id": str,
    "split": "val" | "test",
    "leakage_sources": { "<source_name>": ["<reason>", ...], ... }
  }
Sources with "grades" in the name = LLM grade feature leakage.
Sources with "forecast" in the name = LLM narrative forecast leakage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_LEAKAGE_PATH = _DATA_DIR / "LEAKAGERISK.json"

with _LEAKAGE_PATH.open() as _f:
    _raw = json.load(_f)

EXCLUDE_TEST_LEAKAGE_RISK: bool = bool(_raw.get("exclude_test_leakage_risk", False))

# Build per-source mapping and per-category sets
_leakage_ids_by_source: dict[str, set[str]] = {}
_grade_leakage_ids: set[str] = set()
_all_leakage_ids: set[str] = set()

for _aid, _entry in _raw.items():
    if not isinstance(_entry, dict) or "leakage_sources" not in _entry:
        continue
    _all_leakage_ids.add(_aid)
    for _src in _entry["leakage_sources"]:
        _leakage_ids_by_source.setdefault(_src, set()).add(_aid)
        if "grades" in _src:
            _grade_leakage_ids.add(_aid)

# Maps each leakage source name to the activity IDs affected by it
LEAKAGE_IDS_BY_SOURCE: dict[str, frozenset[str]] = {
    k: frozenset(v) for k, v in _leakage_ids_by_source.items()
}

# Activities with grade-type leakage (any split) — used for feature imputation
GRADE_LEAKAGE_IDS: frozenset[str] = frozenset(_grade_leakage_ids)

# Test-set membership from canonical split CSVs
_split_csvs = [
    _DATA_DIR / "eval_set_sizes" / "overall_ratings_splits.csv",
    _DATA_DIR / "eval_set_sizes" / "outcome_tags_splits.csv",
]
_test_ids: set[str] = set()
for _csv in _split_csvs:
    if _csv.exists():
        _df = pd.read_csv(_csv, dtype={"activity_id": str})
        _test_ids |= set(_df.loc[_df["split"] == "test", "activity_id"])

# Test-set activities with grade leakage (used by G_ outcome tag drop behavior)
TEST_LEAKAGE_RISK_IDS: frozenset[str] = frozenset(_grade_leakage_ids & _test_ids)

# Test-set activities with any leakage (used by A_ prediction swap)
TEST_ANY_LEAKAGE_IDS: frozenset[str] = frozenset(_all_leakage_ids & _test_ids)
