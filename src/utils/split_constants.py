"""
Canonical train/validation/test split configuration.

Single source of truth for all scripts in this project.

  train: activities with start_date <= LATEST_TRAIN_POINT
  val:   LATEST_TRAIN_POINT < start_date <= LATEST_VALIDATION_POINT
  test:  LATEST_VALIDATION_POINT < start_date < TOO_LATE_CUTOFF
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

LATEST_TRAIN_POINT = "2013-02-06"
LATEST_VALIDATION_POINT = "2016-06-06"
TOO_LATE_CUTOFF = "2020-01-01"

# Thesis temporal_splits: exact cutoff dates and ordering
assert (
    LATEST_TRAIN_POINT == "2013-02-06"
), "thesis temporal_splits: train cutoff must be 2013-02-06"
assert (
    LATEST_VALIDATION_POINT == "2016-06-06"
), "thesis temporal_splits: validation cutoff must be 2016-06-06"
assert (
    TOO_LATE_CUTOFF == "2020-01-01"
), "thesis temporal_splits: test end cutoff must be 2020-01-01"
assert pd.to_datetime(LATEST_TRAIN_POINT) < pd.to_datetime(
    LATEST_VALIDATION_POINT
), "train cutoff must precede validation cutoff"
assert pd.to_datetime(LATEST_VALIDATION_POINT) < pd.to_datetime(
    TOO_LATE_CUTOFF
), "validation cutoff must precede test end cutoff"


def split_latest_by_date_with_cutoff(df: pd.DataFrame, date_col: str):
    """
    Canonical date-based train/val/test split.

    Returns (train_idx, val_idx, test_idx) as pd.Index objects.
    """
    n_missing_date = df[date_col].isna().sum()
    if n_missing_date > 0:
        print(
            f"WARNING split_latest_by_date_with_cutoff: {n_missing_date}/{len(df)} rows "
            f"have NaN in '{date_col}' and will be excluded from all splits."
        )
        if n_missing_date > len(df) * 0.10:
            raise ValueError(
                f"split_latest_by_date_with_cutoff: {n_missing_date}/{len(df)} rows "
                f"({n_missing_date/len(df):.0%}) missing '{date_col}' -- too many to silently drop."
            )
    d = df.dropna(subset=[date_col]).sort_values(date_col)
    too_late = d[d[date_col] >= pd.to_datetime(TOO_LATE_CUTOFF)].index
    remaining = d.drop(index=too_late)
    test_idx = remaining[
        remaining[date_col] > pd.to_datetime(LATEST_VALIDATION_POINT)
    ].index
    remaining = remaining.drop(index=test_idx)
    val_idx = remaining[remaining[date_col] > pd.to_datetime(LATEST_TRAIN_POINT)].index
    train_idx = remaining.drop(index=val_idx).index

    # Invariant: splits must be disjoint and non-empty
    assert len(train_idx.intersection(val_idx)) == 0, "BUG: train/val overlap in split"
    assert (
        len(train_idx.intersection(test_idx)) == 0
    ), "BUG: train/test overlap in split"
    assert len(val_idx.intersection(test_idx)) == 0, "BUG: val/test overlap in split"
    assert len(train_idx) > 0, "BUG: train split is empty"
    assert len(val_idx) > 0, "BUG: val split is empty"

    return train_idx, val_idx, test_idx


def assert_split_matches_canonical(
    train_idx,
    val_idx,
    test_idx,
    splits_csv: str | Path | None = None,
) -> None:
    """
    Assert that a computed split is consistent with the canonical split CSV.

    If splits_csv does not exist, issues a warning and skips (safe on first run).
    Compares activity_id sets; the index values must be the activity_ids.
    """
    if splits_csv is None:
        return
    splits_csv = Path(splits_csv)
    if not splits_csv.exists():
        warnings.warn(
            f"Split CSV does not exist: {splits_csv} -- skipping canonical check"
        )
        return

    canon = pd.read_csv(splits_csv, dtype={"activity_id": str})
    canon_train = set(canon.loc[canon["split"] == "train", "activity_id"])
    canon_val = set(canon.loc[canon["split"] == "val", "activity_id"])
    canon_test = set(canon.loc[canon["split"] == "test", "activity_id"])

    def _ids(idx) -> set:
        return set(pd.Index(idx).astype(str))

    errors = []
    for label, computed, canonical in [
        ("train", _ids(train_idx), canon_train),
        ("val", _ids(val_idx), canon_val),
        ("test", _ids(test_idx), canon_test),
    ]:
        if computed != canonical:
            sym = computed.symmetric_difference(canonical)
            errors.append(f"{label}: {len(sym)} mismatched activity_ids")

    if errors:
        raise AssertionError(
            f"Split mismatch vs canonical CSV at {splits_csv}: {'; '.join(errors)}"
        )
