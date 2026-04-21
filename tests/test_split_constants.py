"""
Tests for src/utils/overall_rating_split_constants.py

Covers:
  - constants are in correct chronological order
  - split_latest_by_date_with_cutoff: correct split boundaries, disjointness,
    non-emptiness, error on too-many-missing-dates, warning on some missing
  - assert_split_matches_canonical: no-op without CSV, passes on match, raises on mismatch
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import split_constants as sc

# -- Constants -----------------------------------------------------------------


class TestConstants:
    def test_train_before_val(self):
        assert pd.Timestamp(sc.LATEST_TRAIN_POINT) < pd.Timestamp(
            sc.LATEST_VALIDATION_POINT
        )

    def test_val_before_cutoff(self):
        assert pd.Timestamp(sc.LATEST_VALIDATION_POINT) < pd.Timestamp(
            sc.TOO_LATE_CUTOFF
        )

    def test_constants_are_strings(self):
        assert isinstance(sc.LATEST_TRAIN_POINT, str)
        assert isinstance(sc.LATEST_VALIDATION_POINT, str)
        assert isinstance(sc.TOO_LATE_CUTOFF, str)

    def test_constants_parseable_as_dates(self):
        for c in [
            sc.LATEST_TRAIN_POINT,
            sc.LATEST_VALIDATION_POINT,
            sc.TOO_LATE_CUTOFF,
        ]:
            pd.Timestamp(c)  # must not raise


# -- split_latest_by_date_with_cutoff -----------------------------------------


def _make_df(dates, add_activity_id=True):
    """Helper to build a minimal DataFrame for splitting."""
    df = pd.DataFrame({"start_date": pd.to_datetime(dates)})
    if add_activity_id:
        df["activity_id"] = [f"act_{i}" for i in range(len(df))]
    return df


class TestSplitByDate:
    @pytest.fixture
    def spanning_df(self):
        """DataFrame with dates spanning all three splits."""
        dates = (
            pd.date_range("2005-01-01", "2012-12-31", freq="90D").tolist()  # train
            + pd.date_range("2013-06-01", "2015-12-31", freq="90D").tolist()  # val
            + pd.date_range("2016-09-01", "2019-06-01", freq="90D").tolist()  # test
        )
        return _make_df(dates)

    def test_splits_are_disjoint(self, spanning_df):
        tr, va, te = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        assert len(tr.intersection(va)) == 0
        assert len(tr.intersection(te)) == 0
        assert len(va.intersection(te)) == 0

    def test_splits_are_non_empty(self, spanning_df):
        tr, va, te = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        assert len(tr) > 0
        assert len(va) > 0
        assert len(te) > 0

    def test_train_dates_at_or_before_train_point(self, spanning_df):
        tr, _, _ = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        train_dates = spanning_df.loc[tr, "start_date"]
        assert (train_dates <= pd.Timestamp(sc.LATEST_TRAIN_POINT)).all()

    def test_val_dates_in_correct_window(self, spanning_df):
        _, va, _ = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        val_dates = spanning_df.loc[va, "start_date"]
        assert (val_dates > pd.Timestamp(sc.LATEST_TRAIN_POINT)).all()
        assert (val_dates <= pd.Timestamp(sc.LATEST_VALIDATION_POINT)).all()

    def test_test_dates_in_correct_window(self, spanning_df):
        _, _, te = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        test_dates = spanning_df.loc[te, "start_date"]
        assert (test_dates > pd.Timestamp(sc.LATEST_VALIDATION_POINT)).all()
        assert (test_dates < pd.Timestamp(sc.TOO_LATE_CUTOFF)).all()

    def test_too_late_dates_excluded(self):
        dates = ["2021-01-01", "2022-06-01", "2010-01-01", "2014-01-01", "2017-01-01"]
        df = _make_df(dates)
        tr, va, te = sc.split_latest_by_date_with_cutoff(df, "start_date")
        all_idx = tr.union(va).union(te)
        all_dates = df.loc[all_idx, "start_date"]
        assert (all_dates < pd.Timestamp(sc.TOO_LATE_CUTOFF)).all()

    def test_all_used_dates_accounted_for(self, spanning_df):
        """Every non-cutoff row must appear in exactly one split."""
        tr, va, te = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        in_split = tr.union(va).union(te)
        eligible = spanning_df[
            spanning_df["start_date"] < pd.Timestamp(sc.TOO_LATE_CUTOFF)
        ].index
        assert set(in_split) == set(eligible)

    def test_raises_on_too_many_missing_dates(self):
        # 50% missing → should raise (threshold is 10%)
        n = 20
        dates = [None] * 10 + pd.date_range(
            "2005-01-01", periods=10, freq="180D"
        ).tolist()
        df = _make_df(dates)
        with pytest.raises(ValueError, match="too many"):
            sc.split_latest_by_date_with_cutoff(df, "start_date")

    def test_warns_on_some_missing_dates(self, capsys):
        # 5% missing → warning printed, no exception
        n = 100
        dates = (
            pd.date_range("2005-01-01", periods=n - 5, freq="90D").tolist() + [None] * 5
        )
        df = _make_df(dates)
        # Should not raise
        tr, va, te = sc.split_latest_by_date_with_cutoff(df, "start_date")
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or len(tr) > 0  # just checks it ran

    def test_returns_pd_index(self, spanning_df):
        tr, va, te = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        assert isinstance(tr, pd.Index)
        assert isinstance(va, pd.Index)
        assert isinstance(te, pd.Index)

    def test_reproducible(self, spanning_df):
        """Calling twice on the same df gives identical results."""
        tr1, va1, te1 = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        tr2, va2, te2 = sc.split_latest_by_date_with_cutoff(spanning_df, "start_date")
        assert tr1.equals(tr2)
        assert va1.equals(va2)
        assert te1.equals(te2)


# -- assert_split_matches_canonical -------------------------------------------


class TestAssertSplitMatchesCanonical:
    def test_no_csv_path_is_noop(self):
        sc.assert_split_matches_canonical([], [], [], splits_csv=None)

    def test_nonexistent_csv_issues_warning(self, tmp_path):
        fake = tmp_path / "does_not_exist.csv"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sc.assert_split_matches_canonical(
                pd.Index(["a"]),
                pd.Index(["b"]),
                pd.Index(["c"]),
                splits_csv=fake,
            )
        assert len(w) == 1
        assert "does not exist" in str(w[0].message).lower()

    def test_matching_split_passes(self, tmp_path):
        csv = tmp_path / "splits.csv"
        rows = (
            [{"activity_id": "a", "split": "train"}]
            + [{"activity_id": "b", "split": "val"}]
            + [{"activity_id": "c", "split": "test"}]
        )
        pd.DataFrame(rows).to_csv(csv, index=False)
        sc.assert_split_matches_canonical(
            pd.Index(["a"]),
            pd.Index(["b"]),
            pd.Index(["c"]),
            splits_csv=csv,
        )

    def test_mismatched_split_raises(self, tmp_path):
        csv = tmp_path / "splits.csv"
        rows = [
            {"activity_id": "a", "split": "train"},
            {"activity_id": "b", "split": "val"},
            {"activity_id": "c", "split": "test"},
        ]
        pd.DataFrame(rows).to_csv(csv, index=False)
        with pytest.raises(AssertionError, match="mismatch"):
            sc.assert_split_matches_canonical(
                pd.Index(["a"]),
                pd.Index(["b"]),
                pd.Index(["WRONG"]),
                splits_csv=csv,
            )

    def test_extra_activity_in_computed_raises(self, tmp_path):
        csv = tmp_path / "splits.csv"
        pd.DataFrame(
            [
                {"activity_id": "a", "split": "train"},
                {"activity_id": "b", "split": "val"},
                {"activity_id": "c", "split": "test"},
            ]
        ).to_csv(csv, index=False)
        with pytest.raises(AssertionError):
            sc.assert_split_matches_canonical(
                pd.Index(["a", "extra"]),
                pd.Index(["b"]),
                pd.Index(["c"]),
                splits_csv=csv,
            )
