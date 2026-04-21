"""
Report leakage incidence and prevalence across evaluation sets.

For grade-feature leakage (LLM grades extracted from retrospective section texts),
only the test set is reported, as the val set is not used for thesis conclusions.

For LLM forecast narrative leakage (forecast text contains future outcome information),
both val and test sets are reported so the reader can compare leakage rates across splits.

Run from repo root or any directory; paths are resolved relative to this file.
"""

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
LEAKAGE_PATH = DATA_DIR / "LEAKAGERISK.json"
EVAL_DIR = DATA_DIR / "eval_set_sizes"


def load_leakage() -> dict:
    with LEAKAGE_PATH.open() as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if isinstance(v, dict) and "leakage_sources" in v}


def has_grade(entry: dict) -> bool:
    return any("grade" in src for src in entry["leakage_sources"])


def has_forecast(entry: dict) -> bool:
    return any("forecast" in src for src in entry["leakage_sources"])


def load_split_ids(csv_path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(csv_path, dtype={"activity_id": str})
    df["activity_id"] = df["activity_id"].str.strip()
    splits: dict[str, set[str]] = {}
    for split, grp in df.groupby("split"):
        splits[split] = set(grp["activity_id"])
    return splits


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "N/A"


def report_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def main() -> None:
    entries = load_leakage()

    grade_ids = {k for k, v in entries.items() if has_grade(v)}
    forecast_ids = {k for k, v in entries.items() if has_forecast(v)}
    any_ids = grade_ids | forecast_ids

    # Use overall_ratings splits as the primary reference (same as outcome_tags val/test)
    or_splits = load_split_ids(EVAL_DIR / "overall_ratings_splits.csv")
    val_ids = or_splits.get("val", set())
    test_ids = or_splits.get("test", set())

    report_section("LLM GRADE FEATURE LEAKAGE  (test set only)")
    grade_test = grade_ids & test_ids
    grade_val = grade_ids & val_ids
    eval_ids = val_ids | test_ids
    grade_eval = grade_ids & eval_ids
    print(f"  Test set size              : {len(test_ids)}")
    print(f"  Grade-leakage items (test) : {len(grade_test)}  "
          f"({pct(len(grade_test), len(test_ids))} of test set)")
    print(f"  Grade-leakage items (val)  : {len(grade_val)}")
    print(f"  Grade-leakage overall      : {len(grade_eval)}  "
          f"({pct(len(grade_eval), len(eval_ids))} of val+test combined)")

    report_section("LLM FORECAST NARRATIVE LEAKAGE  (val vs test)")
    forecast_val = forecast_ids & val_ids
    forecast_test = forecast_ids & test_ids
    print(f"  Val  set size              : {len(val_ids)}")
    print(f"  Test set size              : {len(test_ids)}")
    print(f"  Forecast-leakage (val)     : {len(forecast_val)}  "
          f"({pct(len(forecast_val), len(val_ids))} of val set)")
    print(f"  Forecast-leakage (test)    : {len(forecast_test)}  "
          f"({pct(len(forecast_test), len(test_ids))} of test set)")
    report_section("COMBINED  (any leakage, test set)")
    any_test = any_ids & test_ids
    grade_only_test = (grade_ids - forecast_ids) & test_ids
    forecast_only_test = (forecast_ids - grade_ids) & test_ids
    both_test = (grade_ids & forecast_ids) & test_ids
    print(f"  Any leakage                : {len(any_test)}  "
          f"({pct(len(any_test), len(test_ids))} of test set)")
    print(f"    Grade-only               : {len(grade_only_test)}")
    print(f"    Forecast-only            : {len(forecast_only_test)}")
    print(f"    Both grade + forecast    : {len(both_test)}")


if __name__ == "__main__":
    main()
